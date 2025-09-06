# evaluate.py
import os
import re
import stanza
import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
from metaphone import doublemetaphone
from sentence_transformers import SentenceTransformer, util
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
# NEW: Import the POT library for advanced Wasserstein distance
import ot # `uv add pot``, whereas `python import ot`. isn't that funny? isn't that really really funny?
from zss import simple_distance, Node
import numpy as np
from tqdm import tqdm
import torch

# for semantic edit distance metric, which isn't vectorized at the backend (Huh? wuh???)
import multiprocessing
from functools import partial

# ==============================================================================
# METRIC 1: Phonetic & Orthographic Plausibility Score (POPS) - CORRECTED
# ==============================================================================
class POPSScorer:
    """
    Replaces naive "Valid Word Rate" by scoring words on a gradient of plausibility.
    A score of 0 is a perfect dictionary word. A low score indicates a phonetically
    and orthographically plausible neologism. A high score indicates random noise.
    """
    def __init__(self):
        print("Initializing POPS Scorer...")
        # self.dm now correctly holds a reference to the function itself.
        self.dm = doublemetaphone
        self.word_list = set(words.words())
        self.phonetic_map = self._build_phonetic_map()
        print("POPS Scorer ready.")

    def _build_phonetic_map(self):
        phonetic_map = {}
        print("Building phonetic map from dictionary for fast lookup...")
        for word in tqdm(self.word_list):
            try:
                # CORRECT: Call the function directly to get the (primary, secondary) tuple.
                primary, secondary = self.dm(word)

                # Add the word to the map under its primary key.
                if primary:
                    if primary not in phonetic_map:
                        phonetic_map[primary] = []
                    phonetic_map[primary].append(word)
                
                # If a different secondary key exists, add the word there as well.
                if secondary and secondary != primary:
                    if secondary not in phonetic_map:
                        phonetic_map[secondary] = []
                    phonetic_map[secondary].append(word)
            except Exception:
                # Some words (e.g., single letters) might fail in the library
                continue
        return phonetic_map

    def score_word(self, word):
        """Scores a single word based on dictionary presence or phonetic similarity."""
        if not word or not word.isalpha():
            return 0 # Ignore punctuation, numbers, or empty strings

        word_lower = word.lower()
        # Case 1: The word is in the dictionary (perfect score)
        if word_lower in self.word_list:
            return 0

        # Case 2: Word is not in the dictionary; find its phonetic neighbors
        try:
            # CORRECT: Call the function directly on the target word.
            primary_key, secondary_key = self.dm(word_lower)
            
            # Gather all possible phonetic matches from both keys.
            matches = []
            if primary_key in self.phonetic_map:
                matches.extend(self.phonetic_map[primary_key])
            if secondary_key and secondary_key in self.phonetic_map:
                matches.extend(self.phonetic_map[secondary_key])

            if matches:
                # Use set() to get unique matches before calculating distance.
                distances = [edit_distance(word_lower, match) for match in set(matches)]
                return min(distances)
            else:
                # Case 3: No phonetic matches found (high penalty)
                return len(word)
        except Exception:
            # This will catch any unexpected errors from the metaphone library.
            return len(word)

    def score_text(self, text):
        """Calculates the average POPS for a body of text."""
        word_tokens = re.findall(r'\b\w+\b', text)
        if not word_tokens:
            return 0.0

        scores = [self.score_word(word) for word in word_tokens]
        return np.mean(scores)

# ==============================================================================
# METRIC 2: Syntactic Distribution Similarity
# ==============================================================================
# Helper function for multiprocessing - must be defined at the top level of the script
def calculate_ted_row(tree_a, trees_b):
    """Calculates a single row of the cost matrix."""
    return [simple_distance(tree_a, tree_b) for tree_b in trees_b]

class SyntaxAnalyzer:
    def __init__(self):
        print("Initializing Parallelized Syntax Analyzer...")
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', verbose=False)
        print("Syntax Analyzer ready.")

    def _stanza_to_zss(self, stanza_node):
        if not stanza_node.children:
            return Node(stanza_node.label)
        else:
            children = [self._stanza_to_zss(child) for child in stanza_node.children]
            return Node(stanza_node.label, children)

    def analyze_corpora_distance(self, corpus_a_text, corpus_b_text):
        print("Parsing Corpus A...")
        doc_a = self.nlp(corpus_a_text)
        trees_a = [self._stanza_to_zss(s.constituency) for s in tqdm(doc_a.sentences) if hasattr(s, 'constituency')]
        
        print("Parsing Corpus B...")
        doc_b = self.nlp(corpus_b_text)
        trees_b = [self._stanza_to_zss(s.constituency) for s in tqdm(doc_b.sentences) if hasattr(s, 'constituency')]

        if not trees_a or not trees_b:
            print("Warning: One or both corpora produced no parsable trees.")
            return float('nan') #gemini why did you do this >:(

        print(f"Calculating Tree Edit Distance cost matrix for {len(trees_a)} x {len(trees_b)} trees using multiprocessing...")
        
        # Create a pool of worker processes. This will use all available cores.
        with multiprocessing.Pool() as pool:
            # We want to map the `calculate_ted_row` function over each tree in `trees_a`.
            # `partial` is used to "pre-fill" the `trees_b` argument for the worker function.
            worker_func = partial(calculate_ted_row, trees_b=trees_b)
            
            # The `tqdm` will now track the progress of the rows being computed.
            results = list(tqdm(pool.imap(worker_func, trees_a), total=len(trees_a)))
        
        cost_matrix = np.array(results)

        dist_a = np.ones(len(trees_a)) / len(trees_a)
        dist_b = np.ones(len(trees_b)) / len(trees_b)
        
        print("Computing Wasserstein distance...")
        wasserstein_dist = ot.emd2(dist_a, dist_b, cost_matrix)
        return wasserstein_dist

# ==============================================================================
# METRIC 3: Semantic Trajectory Analysis
# ==============================================================================
class SemanticTrajectoryAnalyzer:
    """
    UPGRADED: Analyzes semantic flow by treating each story's trajectory feature vector
    as a single point in a multi-dimensional space.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Initializing Upgraded Semantic Trajectory Analyzer...")
        self.model = SentenceTransformer(model_name)
        print("Semantic Analyzer ready.")

    def _get_story_features(self, story_text):
        sentences = nltk.sent_tokenize(story_text)
        if len(sentences) < 2: return None
        
        embeddings = self.model.encode(sentences) # No need for tensor here
        
        cos_sims = [util.cos_sim(embeddings[i], embeddings[i+1])[0,0] for i in range(len(embeddings)-1)]
        euc_dists = [np.linalg.norm(embeddings[i] - embeddings[i+1]) for i in range(len(embeddings)-1)]

        # This feature vector now represents a single point in our distribution
        return np.array([
            np.mean(cos_sims),          # Mean Coherence
            np.sum(euc_dists),          # Total Drift
            len(sentences)              # Story Length
            # Future features like 'max_excursion' can be added here
        ])

    def analyze_corpora_distance(self, corpus_a_text, corpus_b_text, story_delimiters=["<eos>", "<|endoftext|>"]):
        pattern = '|'.join(map(re.escape, story_delimiters))
        
        stories_a = [s.strip() for s in re.split(pattern, corpus_a_text) if s.strip()]
        stories_b = [s.strip() for s in re.split(pattern, corpus_b_text) if s.strip()]
        
        print(f"Analyzing semantic trajectories of {len(stories_a)} stories from Corpus A...")
        features_a = np.array([f for story in tqdm(stories_a) if (f := self._get_story_features(story)) is not None])
        
        print(f"Analyzing semantic trajectories of {len(stories_b)} stories from Corpus B...")
        features_b = np.array([f for story in tqdm(stories_b) if (f := self._get_story_features(story)) is not None])

        if features_a.shape[0] == 0 or features_b.shape[0] == 0:
            print("Warning: One or both corpora had no valid stories. Cannot compute distance.")
            return float('nan')

        # Here, the ground distance is the Euclidean distance between the feature vectors
        print(f"Calculating Euclidean distance cost matrix for {len(features_a)} x {len(features_b)} feature vectors...")
        cost_matrix = cdist(features_a, features_b, 'euclidean')

        dist_a = np.ones(len(features_a)) / len(features_a)
        dist_b = np.ones(len(features_b)) / len(features_b)

        print("Computing Wasserstein distance...")
        wasserstein_dist = ot.emd2(dist_a, dist_b, cost_matrix)
        return wasserstein_dist

# ==============================================================================
# --- Main Execution Logic ---
# ==============================================================================
def main():
        # --- SETUP: Load your generated text and a ground-truth validation set ---
    # For demonstration, we'll use placeholder text.
    # In a real run, you would load your files like this:
    # with open('path/to/your_model_output.txt', 'r') as f:
    #     generated_text = f.read()
    # with open('path/to/tinystories_validation.txt', 'r') as f:
    #     ground_truth_text = f.read()

    
    # --- One-time Downloads for NLTK and Stanza ---
    print("Downloading necessary NLP models...")
    nltk.download('words', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    stanza.download('en', verbose=False) # Download the English model for Stanza
    print("Downloads complete.")


    ground_truth_text = """
    <|endoftext|>u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
    <|endoftext|>
    Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
    Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
    Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
    They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
    <|endoftext|>


    Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
    "Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
    "No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
    Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
    Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
    "Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
    "Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
    "We're okay, Mommy. But our toys are broken," Lily said.
    "I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
    Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
    "Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
    Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
    They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
    Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
    <|endoftext|>

    Once 
    """

    generated_text = """
    <eos>
    Once upon a time, there was a shiny dog named Spot. Spot loved to dig to get the ball and ran home. He paid through the other feet scared.
    One day, Spot went to the park to play. He saw a big tree. He was very excited. He opened his scarf to play. He wanted to hear the sound find food. He followed the nice to panic in the pock.
    "Thank you for a long join. If you want to see your friend too. They are very happy." But then, the fork saw something spiking. It saw the farmer.
    The farmer wanted to joke to carrots and
    <eos>
    Once upon a time, there was a little boy named Tim. Tim was always very tired and had lots of more. On the way home, Tim, something unexpected happened.
    Tim saw a girl named Spot was not nice to hide. The trees man out of the town when it helped Tim, it was cute candy. In the park was in many things, naps and found a new toy car. Tim had a few dragon sticker that before he had to find the owner a vegetable.
    Tim and his friends were happy to have the thing to the newspaper. Finally, they had many fun, and Tim bec
    <eos>
    Once upon a time, in a small town, there lived a room looked for thin arms, ' don't want to buy foolish one on a haircuts together. Ella was scared and she had many things castle. The raven said, Moral, is my fuet in these waste, for you to play with!"
    One day, Mia took his own her room and her window starting the cookies about how to play with a big aeroplane so. The fire stucks were waiting for the new songs enor finds. They put on the ground and broke a new pictures with return.
    "Mom, was now what happens, d
    <eos>
    Once upon a time, there was a cat. The cat wanted to spin every day. At the cat and found a way to messy cat.
    The cat said, "Dor thank you" to, the cat was safe. The cat looked scary and wet. The cat said, "You're welcome, cat listen us, it is a young cat." The cat and the cat said, "You are the cat."
    Everyone and said, "It need to catch that can make you feel safe."
    The cat and his mom for being scared, and they shared the cat ran away from the cat. The cat ate the cat was happy, but not be brave. So she wanted
    <eos>
    Once upon a time, there was a very special duck on the ground. The frog would feel tired, but he did not know if it was okay.
    One day, a little girl wanted to play. She was very excited but very hard. So, she asked Max, "Can we eat the cake looking for a meeting alieve when he find it." The police was so happy to find him. Let's look turns red and soon, it would laugh with a heavy found it.
    As the flower, they do not an after the tent. The clumsy took her needle tasted a fun trick. It was healthy and he was prou
    """
    
    # --- 1. Calculate POPS ---
    print("\n--- Evaluating POPS ---")
    pops_scorer = POPSScorer()
    generated_pops = pops_scorer.score_text(generated_text)
    ground_truth_pops = pops_scorer.score_text(ground_truth_text)
    print(f"Ground Truth Average Word Plausibility Score (POPS): {ground_truth_pops:.4f}")
    print(f"Generated Text Average Word Plausibility Score (POPS): {generated_pops:.4f} (Lower is better)")

    # --- 2. UPGRADED: Calculate Syntactic Wasserstein Distance ---
    print("\n--- Evaluating Syntax ---")
    syntax_analyzer = SyntaxAnalyzer()
    syntactic_distance = syntax_analyzer.analyze_corpora_distance(ground_truth_text, generated_text)
    print(f"\nSyntactic Distribution Distance (Wasserstein over TED): {syntactic_distance:.4f} (Lower is better)")
    
    # --- 3. UPGRADED: Calculate Semantic Trajectory Wasserstein Distance ---
    print("\n--- Evaluating Semantics ---")
    semantic_analyzer = SemanticTrajectoryAnalyzer()
    # For semantics, we use the ORIGINAL text to preserve story boundaries
    semantic_distance = semantic_analyzer.analyze_corpora_distance(ground_truth_text, generated_text)
    print(f"\nSemantic Trajectory Distance (Wasserstein over Feature Vectors): {semantic_distance:.4f} (Lower is better)")

if __name__ == '__main__':
    main()

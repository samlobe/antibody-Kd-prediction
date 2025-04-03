# esm_utils.py

from esm.pretrained import ESM3_sm_open_v0
from esm.sdk import client
from esm.sdk.forge import ESM3InferenceClient, ESMProtein, LogitsConfig
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.constants import esm3 as C
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import torch
import time
import copy
import sys
import os
import threading
import csv
import concurrent.futures


def _limit_api_rate(calls_per_second: float = 5.0):
    """
    Simple helper to avoid exceeding ~5 calls/second
    by sleeping for the necessary fraction of a second.
    If you run multiple threads, each one will do this
    after finishing an API call, effectively bounding
    your total concurrency rate.
    """
    time.sleep(1.0 / calls_per_second)

class ESMUtils:
    """This is a class for helpful utility methods to interface with ESM.

    Parameters
    ----------
    esm3_api_token : str
        Your ESM API token.
    model_name : str, optional
        The name of the ESM model you want to use, by default "esm3-medium-multimer-2024-08"
    """
    protein_encoded = None
    sequence_vocab = C.SEQUENCE_VOCAB

    def __init__(self, esm3_api_token: str, model_name: str="esm3-medium-multimer-2024-09"):
        if model_name == "local-cpu":
            self.model = ESM3_sm_open_v0(device="cpu")       
        elif model_name == "local-gpu":
            self.model = ESM3_sm_open_v0(device="cuda")
        else:
            # Private ESM model hosted at meta
            self.model: ESM3InferenceClient = client(
                model=model_name, 
                token=esm3_api_token
            )

    @staticmethod
    def get_protein_from_pdb(pdb_path: str, is_protein_complex: bool) -> Tuple[ESMProtein, ProteinComplex]:
        """Returns an ESMProtein from a PDB file.

        Parameters
        ----------
        pdb_path : str
            The filepath to the PDB file.
        is_protein_complex : bool
            True if the PDB file is a protein complex (multiple structures).

        Returns
        -------
        Tuple[ESMProtein, ProteinComplex]
            The ESMProtein and (optionally) ProteinComplex data objects of the PDB file.
        """
        protein_complex = None
        if is_protein_complex:
            protein_complex = ProteinComplex.from_pdb(path=pdb_path)
            protein = ESMProtein.from_protein_complex(protein_complex=protein_complex)
        else:
            protein = ESMProtein.from_pdb(path=pdb_path)
        protein.potential_sequence_of_concern = True
        return protein, protein_complex

    def get_logits(self, protein: ESMProtein, 
                   return_embeddings: bool = False
                   ) -> Tuple[torch.Tensor, float]:
        """
        Return the logits of the ESMProtein and the (average) log-likelihood.

        Parameters
        ----------
        protein : ESMProtein
            The protein (or protein complex) to generate logits from.
        return_embeddings : bool
            Whether to return embeddings from the model as well.

        Returns
        -------
        logits_tensor : torch.Tensor
            Tensor of shape (seq_len-2, vocab_size).
        avg_log_likelihood : float
            The average log-likelihood.
        """
        # 1) Encode the protein
        self.protein_encoded = self.model.encode(input=protein)

        # 2) Obtain the logits
        logits_output = self.model.logits(
            input=self.protein_encoded,
            config=LogitsConfig(sequence=True, return_embeddings=return_embeddings),
        )
        
        # 3) Slice out BOS/EOS if you do not want them
        # For private ESM3InferenceClient, the shape is [seq_len, vocab_size]
        logits_tensor = logits_output.logits.sequence[1:-1] # ignoring BOS and EOS tokens

        # 4) Build the target indices from the actual sequence (length = L).
        # turn protein.sequence into a list of chars
        protein_sequence_list = list(protein.sequence)
        # convert any '_' tokens to '<mask>'
        protein_sequence_list = ['<mask>' if aa == '_' else aa for aa in protein_sequence_list]
        targets_list = [self.sequence_vocab.index(aa) for aa in protein_sequence_list]
        targets_tensor = torch.tensor(targets_list, dtype=torch.long, device=logits_tensor.device)
        # shape => (L,)

        # 5) Check length match
        assert logits_tensor.shape[0] == targets_tensor.shape[0], (
            f"Mismatched shapes: logits={logits_tensor.shape}, targets={targets_tensor.shape}"
        )

        # # 6) Compute cross-entropy (avg)
        # nll_loss = F.cross_entropy(logits_tensor, targets_tensor, reduction='mean')

        # # 7) Convert negative log-likelihood to "average log-likelihood"
        # avg_log_likelihood = -nll_loss.item()

        nll_loss = F.cross_entropy(logits_tensor, targets_tensor, reduction='none')
        avg_log_likelihood = -nll_loss.mean().item()

        # Rate-limit to avoid blowing your ESM usage
        _limit_api_rate(calls_per_second=5.0)

        return logits_tensor.detach().cpu(), nll_loss, avg_log_likelihood

    def compute_token_log_likelihoods(self, protein: 'ESMProtein',
                                        identifier: str,
                                        checkpoint_file: str,
                                        mask_token: int = 32,
                                        chainbreak_token: int = 31,
                                        calls_per_second: float = 5.0,
                                        max_retries: int = 3,
                                        max_workers: int = 8) -> dict:
        """
        Compute token-level log likelihoods for the given protein.
        Checkpointing is done in a CSV file whose rows are specific to this antibody (identifier).
        Returns a dictionary mapping token indices to log likelihood values.
        """
        import csv, copy, time, concurrent.futures, numpy as np, torch
        from esm.sdk.forge import LogitsConfig

        def _limit_api_rate(calls_per_second: float = 5.0):
            time.sleep(1.0 / calls_per_second)

        # --- Load existing checkpoint for this antibody ---
        checkpoint = {}
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)  # header row
                    id_idx = header.index("identifier")
                    token_idx = header.index("token_index")
                    ll_idx = header.index("log_likelihood")
                    for row in reader:
                        if row[id_idx] == identifier:
                            idx = int(row[token_idx])
                            checkpoint[idx] = float(row[ll_idx])
            except Exception as e:
                print(f"[WARNING] Could not properly read checkpoint file {checkpoint_file}: {e}")

        # --- Encode protein once ---
        encoded = self.model.encode(input=protein)
        encoded.potential_sequence_of_concern = True
        if encoded.sequence.dim() == 2:
            orig_seq = encoded.sequence[0].clone()
        else:
            orig_seq = encoded.sequence.clone()
        L = orig_seq.shape[0]

        # Process positions 1 to L-2 (skipping BOS/EOS)
        token_indices = [i for i in range(1, L - 1)
                        if orig_seq[i].item() != chainbreak_token and i not in checkpoint]

        # Create (or ensure) the checkpoint file has a header.
        import threading
        file_lock = threading.Lock()
        if not os.path.exists(checkpoint_file):
            # create the file (including dirs)
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            with file_lock, open(checkpoint_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["identifier", "token_index", "log_likelihood"])

        def compute_for_token(i: int) -> tuple:
            attempts = 0
            while attempts < max_retries:
                try:
                    masked_seq = orig_seq.clone()
                    masked_seq[i] = mask_token
                    encoded_masked = copy.deepcopy(encoded)
                    if encoded_masked.sequence.dim() == 2:
                        encoded_masked.sequence[0] = masked_seq
                    else:
                        encoded_masked.sequence = masked_seq
                    logits_output = self.model.logits(
                        input=encoded_masked,
                        config=LogitsConfig(sequence=True, return_embeddings=False)
                    )
                    logits_tensor = logits_output.logits.sequence  # shape: [L, vocab_size]
                    token_logits = logits_tensor[i, :]  # note: adjust if necessary
                    log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                    true_token = orig_seq[i].item()
                    token_log_prob = log_probs[true_token].item()
                    _limit_api_rate(calls_per_second)
                    return i, token_log_prob
                except Exception as e:
                    attempts += 1
                    backoff = 2 ** attempts
                    print(f"Token {i} (identifier {identifier}): attempt {attempts} failed with error {e}. Retrying in {backoff} s...")
                    time.sleep(backoff)
            print(f"Token {i} (identifier {identifier}): max retries reached. Recording np.nan")
            return i, np.nan

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(compute_for_token, i): i for i in token_indices}
            for future in concurrent.futures.as_completed(future_to_index):
                idx, log_likelihood = future.result()
                results[idx] = log_likelihood
                with file_lock, open(checkpoint_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([identifier, idx, log_likelihood])
                print(f"Computed token {idx} for {identifier}: log likelihood = {log_likelihood}")
        checkpoint.update(results)
        return dict(sorted(checkpoint.items()))

    @staticmethod
    def clean_pae_matrix(pae_matrix: Tensor, protein: ESMProtein) -> np.ndarray:
        """Removes BOS/EOS tokens and `|` delimiters from the PAE matrix

        Parameters
        ----------
        pae_matrix : Tensor
            The PAE matrix from the decoded ESMProtein.
        protein : ESMProtein
            The ESMProtein containing the entire sequence.

        Returns
        -------
        numpy.ndarray
            The cleaned PAE matrix.
        """
        pae_matrix = pae_matrix.cpu().detach().numpy()[0]

        msg = f"Shape Misalignment: PAE of shape {pae_matrix.shape} does not match expected tokens."
        assert pae_matrix.shape[0] == len(protein.sequence) + 2, msg
        assert pae_matrix.shape[0] == pae_matrix.shape[1], (
            f"PAE is not square: {pae_matrix.shape}"
        )

        pipe_indexes = [i + 1 for i, aa in enumerate(protein.sequence) if aa == "|"]
        non_aa_indexes = [0, *pipe_indexes, -1]

        pae_matrix = np.delete(pae_matrix, obj=non_aa_indexes, axis=1)
        pae_matrix = np.delete(pae_matrix, obj=non_aa_indexes, axis=0)

        return pae_matrix

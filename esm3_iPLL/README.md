This is a work in progress. We are planning to make it user friendly and to make this available on the cloud for easy use.

Current Procedure:
- Put a pdb of your complex in the data directory, along with a fasta file of sequences to test.
- Make sure you have access to the esm3-large-multimer-2024-09 model on EvolutionaryScale's forge, and you have esm installed locally.
- Set your ESM3 API KEY as an environmental variable called ESM3_KEY.
- Install mdtraj (conda install -c conda-forge mdtraj) and use `python get_contact_mask.py` to make an interface mask for your protein complex.
- Use `python single_pass_LLs_parallel.py` to output LLs of all residues to the data folder; first adjust the file names and the number of CPU processors to make API calls in parallel so that you don't hit rate limits.
- Then use `python convert_PLLs_to_iPLL.py` to get the iPLLs, which may predict useful properties for your experiments.

A cutoff 1.0 nm was deemed to be ideal to define interface residues (distances between alpha carbons); see ../images/esm3-large-multimer-2024-09_cutoff_sensitivity.png
# Diffusion-Based Reranking for Conversational Memory Retrieval

This project explores whether a conditional denoising diffusion model can improve the retrieval of user-specific memories from large conversational memory banks. By reranking the top-$K$ candidate facts retrieved by a standard SBERT encoder (all-MiniLM-L6-v2), the diffusion model learns a continuous distribution over correct answers. We evaluate two diffusion strategies: prototype generation (Approach A) and denoising-score reranking (Approach B).

## How to Run the Project

**We highly recommend running this project in Google Colab with a GPU.** 
The full pipeline (including data generation, SBERT embedding, training the diffusion models, and evaluating the baselines) is computationally intensive and takes **approximately 7 hours to finish running** from start to finish.

1. Open Google Colab and ensure your runtime type is set to **GPU** (e.g., T4, V100, or A100).
2. Upload the `updated_159.ipynb` notebook to Colab.
3. Run the first cell to set up the checkpointing system. It will ask for permission to mount your Google Drive. We recommend granting this permission so your progress is saved.
4. Run the remaining cells in the notebook. All necessary dependencies (such as `sentence-transformers`, `datasets`, `rank_bm25`, and `tqdm`) will be installed automatically by the notebook.
5. The notebook will automatically generate and save output evaluation charts as `.png` files in the current working directory.

## Model Weights and Checkpointing

Because the training process takes several hours, the notebook includes a robust checkpointing system.

- **Google Drive Integration:** If you allow the notebook to mount your Google Drive, it will automatically save model weights (`.pt` files) and training history to `/content/drive/MyDrive/cs159_memory_diffusion_checkpoints/`. 
- **Resuming Execution:** If your Colab session disconnects or times out, you can simply reconnect, mount your Drive again, and run the notebook. The script will automatically detect the saved `.pt` files in your Drive, load the pretrained weights, and skip the training loops that have already completed.
- **Local Fallback:** If you choose not to mount Google Drive or if you run the script locally, checkpoints will be saved to a local folder named `cs159_memory_diffusion_checkpoints/` in your current working directory.

## File Structure
- `updated_159.ipynb`: The primary, Colab-ready notebook containing the entire pipeline (data corruption, embedding, diffusion training, evaluation, and plotting).
- `paper/main.tex`: The LaTeX source code for the final report describing the methodology, statistical analysis, and results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

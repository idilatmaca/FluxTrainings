# FluxTrainings

## Setup

Clone the repository:
```bash
git clone https://github.com/idilatmaca/FluxTrainings.git

cd diffusers
pip install .

cd examples/dreambooth
pip install -r requirements_flux.txt
```

Navigate to the training project:
```bash
cd research_projects/flux_lora_quantaztion
```b


Embedding Computation

Run the following command to compute embeddings:

python compute_embeddings.py \
  --max_sequence_length 77 \
  --output_path custom_embeddings.parquet

Training

To start training, run:

bash train.sh


from inside research_projects/flux_lora_quantaztion.

Testing

After training, you can test the model with:

python test_mode.py



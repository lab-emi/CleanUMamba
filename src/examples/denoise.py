import argparse
import os
import torch
from torchinfo import summary
from tqdm import tqdm
from scipy.io.wavfile import write as wavwrite

from src.examples.loading_pretrained_models import load_pretrained_CleanUMamba
from src.network.network import Net
from src.util.dataset import load_NoisyDataset
from src.util.util import sampling


def denoise(checkpoint_path, input_directory, output_directory=None):
    """
    Denoise audio

    Parameters:
    checkpoint_path (str):          path to the checkpoint to load
    input_directory (str):          denoise noisy audio from this path
    output_directory (str/None):    save generated speeches to this path if provided
    """
    sample_rate = 16000


    # load checkpoint
    model = load_pretrained_CleanUMamba(checkpoint_path)
    model.eval()

    # Print the model summery
    summary(model, input_size=(1, 1, int(1600)), dtypes=[torch.float32])

    # get output directory ready
    if output_directory is not None and output_directory != '':
        print(f"output_directory: {output_directory}")

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)


    dataloader = load_NoisyDataset(
        folder=input_directory,
        sample_rate=sample_rate,
        batch_size=1,
        num_gpus=1
    )


    # inference
    all_cleaned_audio = []
    all_noisy_audio = []
    sortkey = lambda name: '_'.join(name.split('/')[-1].split('_')[1:])
    for noisy_audio1, noisy_audio, _, filename in tqdm(dataloader):

        noisy_audio = noisy_audio.cuda()

        # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #     with record_function("model_inference"):
        generated_audio = sampling(model, noisy_audio, split_sampling=False, block_size=160000)

        if output_directory:
            wavwrite(os.path.join(output_directory, f'enhanced_{filename[0]}'),
                     sample_rate,
                     generated_audio[0].squeeze().cpu().numpy())

        all_noisy_audio.append(noisy_audio1[0].squeeze().cpu().numpy())
        all_cleaned_audio.append(generated_audio[0].squeeze().cpu().numpy())

    # prof.export_chrome_trace("trace.json")

    return all_noisy_audio, all_cleaned_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', default='../../checkpoints/pruned/CleanUMamba-3N-E8_pruned-5M.pkl',
                        help='Path to the checkpoint')

    parser.add_argument('-i', '--input_directory', type=str, default='/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge/datasets/test_set/synthetic/no_reverb/noisy',
                        help='Input folder path to load noisy only data from')
    parser.add_argument('-o', '--output_directory', type=str, default='../../checkpoints/pruned/CleanUMamba-3N-E8_pruned-5M_denoised_audio/',
                        help='Output folder path to save the audio to')
    args = parser.parse_args()


    denoise(args.checkpoint_path, args.input_directory, args.output_directory)

    
        

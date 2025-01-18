from model import SeResNext50_Unet_Loc
import argparse
from torchvision.io import read_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segemntation Model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    model = SeResNext50_Unet_Loc().cuda()
    image_tensor = read_image(args.image_path).unsqueeze(0).cuda()
    output = model(image_tensor)
    print(output)
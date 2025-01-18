from model import SeResNext50_Unet_Loc
import argparse
import rasterio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segemntation Model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    model = SeResNext50_Unet_Loc().cuda()
    with rasterio.open(args.image_path) as src:
        img_array = src.read()
    image_tensor = torch.from_numpy(img_array).cuda()
    return img_array
    output = model(image_tensor)
    print(output)
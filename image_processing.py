import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import traceback
from scipy.ndimage import gaussian_filter
import os
from skimage import exposure


def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    return image.to(device), Image.open(image_path).convert('RGB')


def generate_depth_map(image_tensor, model, model_name):
    with torch.no_grad():
        if "DepthAnythingV2" in model_name:
            raw_img = np.array(image_tensor.squeeze().permute(1, 2, 0).cpu())
            depth_map = model.infer_image(raw_img)
            depth_map = cv2.resize(depth_map, (raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            outputs = model(image_tensor)
            if "Zoe" in model_name:
                depth_map = outputs['metric_depth']
            elif "MiDaS" in model_name:
                prediction = torch.nn.functional.interpolate(
                    outputs.unsqueeze(1),
                    size=image_tensor.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth_map = prediction.cpu().numpy()
            else:
                depth_map = outputs
            depth_map = transforms.functional.resize(depth_map, (
                image_tensor.shape[2], image_tensor.shape[3])).squeeze().cpu().numpy()
    return depth_map


def save_depth_map(depth_map, output_path, display_mode):
    try:
        depth_map_np = depth_map.squeeze() if torch.is_tensor(depth_map) else depth_map
        depth_map_np = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())

        if "Inverted" in display_mode:
            depth_map_np = 1 - depth_map_np

        if "Colored" in display_mode:
            depth_map_img = plt.cm.viridis(depth_map_np)[:, :, :3]  # Use a colormap
            depth_map_img = (depth_map_img * 255).astype(np.uint8)
            plt.imsave(output_path, depth_map_img)
        else:
            if display_mode == "16bit":
                depth_map_16bit = (depth_map_np * 65535).astype(np.uint16)
                plt.imsave(output_path, depth_map_16bit, cmap='gray')
            elif display_mode == "8bit":
                depth_map_8bit = (depth_map_np * 255).astype('uint8')
                plt.imsave(output_path, depth_map_8bit, cmap='gray')
            else:
                plt.imsave(output_path, depth_map_np, cmap='gray')
    except Exception as e:
        print(f"Error in save_depth_map: {e}")


def generate_high_quality_depth_map(image, model, model_name, log_queue, assets_folder, display_mode, output_path,
                                    tile_sizes=None, filter_strength=0.998, sigma_blur=20, sigma_difference=40,
                                    difference_multiplier=5, gaussian_sigma=2):
    try:
        if tile_sizes is None:
            tile_sizes = [[4, 4]]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        log_queue.put("Generating low resolution depth map...\n")

        # Generate low resolution depth map
        if "DepthAnythingV2" in model_name:
            raw_img = np.array(image)
            low_res_depth = model.infer_image(raw_img)
        elif "Zoe" in model_name:
            output = model(image_tensor)
            if 'metric_depth' in output:
                low_res_depth = output['metric_depth'].squeeze().cpu().detach().numpy()
            else:
                raise ValueError("Expected key 'metric_depth' not found in model output")
        else:
            low_res_depth = model(image_tensor).squeeze().cpu().detach().numpy()

        # Normalize low resolution depth map
        low_res_scaled_depth = 2 ** 16 - (low_res_depth - np.min(low_res_depth)) * 2 ** 16 / (
                np.max(low_res_depth) - np.min(low_res_depth))
        low_res_depth_map_image = Image.fromarray((0.999 * low_res_scaled_depth).astype("uint16"))
        low_res_depth_map_image.save(os.path.join(assets_folder, '16bit_low.png'))
        log_queue.put("Low resolution depth map generated.\n")

        im = np.asarray(image)
        filters = []

        # Generate filters for different tile sizes
        for tile_size in tile_sizes:
            num_x, num_y = tile_size
            M, N = im.shape[0] // num_x, im.shape[1] // num_y

            filter_dict = {f'{direction}_filter': np.zeros((M, N)) for direction in
                           ['right', 'left', 'top', 'bottom', 'top_right', 'top_left', 'bottom_right', 'bottom_left']}
            filter_dict['filter'] = np.zeros((M, N))

            for i in range(M):
                for j in range(N):
                    x_value = filter_strength * np.cos((abs(M / 2 - i) / M) * np.pi) ** 2
                    y_value = filter_strength * np.cos((abs(N / 2 - j) / N) * np.pi) ** 2

                    filter_dict['right_filter'][i, j] = x_value if j > N / 2 else x_value * y_value
                    filter_dict['left_filter'][i, j] = x_value if j < N / 2 else x_value * y_value
                    filter_dict['top_filter'][i, j] = y_value if i < M / 2 else x_value * y_value
                    filter_dict['bottom_filter'][i, j] = y_value if i > M / 2 else x_value * y_value

                    filter_dict['top_right_filter'][i, j] = (filter_strength if (j > N / 2 and i < M / 2) else
                                                             x_value if j > N / 2 else
                                                             y_value if i < M / 2 else x_value * y_value)

                    filter_dict['top_left_filter'][i, j] = (filter_strength if (j < N / 2 and i < M / 2) else
                                                            x_value if j < N / 2 else
                                                            y_value if i < M / 2 else x_value * y_value)

                    filter_dict['bottom_right_filter'][i, j] = (filter_strength if (j > N / 2 and i > M / 2) else
                                                                x_value if j > N / 2 else
                                                                y_value if i > M / 2 else x_value * y_value)

                    filter_dict['bottom_left_filter'][i, j] = (filter_strength if (j < N / 2 and i > M / 2) else
                                                               x_value if j < N / 2 else
                                                               y_value if i > M / 2 else x_value * y_value)

                    filter_dict['filter'][i, j] = x_value * y_value

            filters.append(filter_dict)

            for filter in filter_dict:
                filter_image = Image.fromarray((filter_dict[filter] * 2 ** 16).astype("uint16"))
                filter_image.save(os.path.join(assets_folder, f'mask_{filter}_{num_x}_{num_y}.png'))

        compiled_tiles_list = []

        # Process tiles
        for i, filter_dict in enumerate(filters):
            num_x, num_y = tile_sizes[i]
            M, N = im.shape[0] // num_x, im.shape[1] // num_y
            compiled_tiles = np.zeros((im.shape[0], im.shape[1]))

            x_coords = list(range(0, im.shape[0], im.shape[0] // num_x))[:num_x]
            y_coords = list(range(0, im.shape[1], im.shape[1] // num_y))[:num_y]
            x_coords_between = list(range((im.shape[0] // num_x) // 2, im.shape[0], im.shape[0] // num_x))[:num_x - 1]
            y_coords_between = list(range((im.shape[1] // num_y) // 2, im.shape[1], im.shape[1] // num_y))[:num_y - 1]

            x_coords_all = x_coords + x_coords_between
            y_coords_all = y_coords + y_coords_between

            for x in x_coords_all:
                for y in y_coords_all:
                    try:
                        x_end, y_end = min(x + M, im.shape[0]), min(y + N, im.shape[1])
                        tile = np.array(im[x:x_end, y:y_end])
                        if tile.ndim == 2:
                            tile = np.stack([tile] * 3, axis=-1)
                        elif tile.shape[2] == 1:
                            tile = np.concatenate([tile] * 3, axis=-1)

                        if "DepthAnythingV2" in model_name:
                            depth = model.infer_image(tile)
                        elif "Zoe" in model_name:
                            depth = model(transforms.ToTensor()(Image.fromarray(np.uint8(tile))).unsqueeze(0).to(device))['metric_depth'].squeeze().cpu().detach().numpy()
                        else:
                            depth = model(transforms.ToTensor()(Image.fromarray(np.uint8(tile))).unsqueeze(0).to(device)).squeeze().cpu().detach().numpy()

                        scaled_depth = 2 ** 16 - (depth - np.min(depth)) * 2 ** 16 / (np.max(depth) - np.min(depth))

                        selected_filter = get_selected_filter(i, x, y, x_coords_all, y_coords_all, filters)

                        filter_slice = selected_filter[:x_end - x, :y_end - y]
                        low_res_slice = low_res_scaled_depth[x:x_end, y:y_end]

                        if filter_slice.shape == scaled_depth[:filter_slice.shape[0], :filter_slice.shape[1]].shape == low_res_slice.shape:
                            compiled_tiles[x:x_end, y:y_end] += filter_slice * (
                                    np.mean(low_res_slice) + np.std(low_res_slice) * ((scaled_depth[:filter_slice.shape[0], :filter_slice.shape[1]] - np.mean(scaled_depth[:filter_slice.shape[0], :filter_slice.shape[1]])) / np.std(scaled_depth[:filter_slice.shape[0], :filter_slice.shape[1]])))

                        log_queue.put(f"Processed tile ({x}, {y}) to ({x_end}, {y_end}) for tile size {tile_sizes[i]}.\n")
                    except Exception as e:
                        log_queue.put(f"Error processing tile ({x}, {y}) to ({x_end}, {y_end}): {str(e)}\n")
                        log_queue.put(traceback.format_exc() + "\n")

            compiled_tiles[compiled_tiles < 0] = 0
            compiled_tiles = np.nan_to_num(compiled_tiles)
            compiled_tiles_list.append(compiled_tiles)

            max_val = np.max(compiled_tiles)
            if max_val > 0:
                compiled_tiles = 2 ** 16 * 0.999 * compiled_tiles / max_val
            compiled_tiles = compiled_tiles.astype("uint16")
            tiled_depth_map = Image.fromarray(compiled_tiles)
            tiled_depth_map.save(os.path.join(assets_folder, f'tiled_depth_{tile_sizes[i]}.png'))
            tiled_depth_map.save(os.path.join('maps', f'tiled_depth_{tile_sizes[i]}.png'))
            log_queue.put(f"Tiled depth map for tile size {tile_sizes[i]} saved.\n")

        grey_im = np.mean(im, axis=2)
        tiles_blur = gaussian_filter(grey_im, sigma=sigma_blur)
        tiles_difference = tiles_blur - grey_im
        tiles_difference = tiles_difference / np.max(tiles_difference)
        tiles_difference = gaussian_filter(tiles_difference, sigma=sigma_difference)
        tiles_difference *= difference_multiplier
        tiles_difference = np.clip(tiles_difference, 0, 0.999)

        mask_image = Image.fromarray((tiles_difference * 2 ** 16).astype("uint16"))
        mask_image.save(os.path.join(assets_folder, 'mask_image.png'))
        log_queue.put("Mask image saved.\n")

        min_shape = np.min([compiled_tiles_list[0].shape, compiled_tiles_list[1].shape, low_res_scaled_depth.shape], axis=0)
        tiles_difference = tiles_difference[:min_shape[0], :min_shape[1]]
        combined_result = (tiles_difference * compiled_tiles_list[1][:min_shape[0], :min_shape[1]] + (
                1 - tiles_difference) * ((compiled_tiles_list[0][:min_shape[0], :min_shape[1]] + low_res_scaled_depth[:min_shape[0], :min_shape[1]]) / 2)) / 2
        combined_result = np.nan_to_num(combined_result)

        # Apply histogram equalization to improve the gradient
        combined_result = exposure.equalize_hist(combined_result)

        # Reduce over-smoothing by applying a smaller sigma value for Gaussian filter
        combined_result = gaussian_filter(combined_result, sigma=gaussian_sigma)

        max_val = np.max(combined_result)
        if max_val > 0:
            combined_result = 2 ** 16 * 0.999 * combined_result / max_val
        combined_result = combined_result.astype("uint16")

        save_depth_map(combined_result, output_path, display_mode)
        log_queue.put("High quality depth map generation complete.\n")

        for file in os.listdir(assets_folder):
            file_path = os.path.join(assets_folder, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                log_queue.put(f'Failed to delete {file_path}. Reason: {e}')

        return combined_result
    except Exception as e:
        log_queue.put(f"Error in generate_high_quality_depth_map: {str(e)}\n")
        log_queue.put(traceback.format_exc() + "\n")


def get_selected_filter(i, x, y, x_coords_all, y_coords_all, filters):
    if y == min(y_coords_all) and x == min(x_coords_all):
        selected_filter = filters[i]['top_left_filter']
    elif y == min(y_coords_all) and x == max(x_coords_all):
        selected_filter = filters[i]['bottom_left_filter']
    elif y == max(y_coords_all) and x == min(x_coords_all):
        selected_filter = filters[i]['top_right_filter']
    elif y == max(y_coords_all) and x == max(x_coords_all):
        selected_filter = filters[i]['bottom_right_filter']
    elif y == min(y_coords_all):
        selected_filter = filters[i]['left_filter']
    elif y == max(y_coords_all):
        selected_filter = filters[i]['right_filter']
    elif x == min(x_coords_all):
        selected_filter = filters[i]['top_filter']
    elif x == max(x_coords_all):
        selected_filter = filters[i]['bottom_filter']
    else:
        selected_filter = filters[i]['filter']
    return selected_filter

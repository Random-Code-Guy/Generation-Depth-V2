import os
import traceback
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    return image.to(device), Image.open(image_path).convert('RGB')


def save_depth_map(gui, depth_map, output_path, display_mode, color_map):
    try:
        depth_map_np = depth_map.squeeze() if torch.is_tensor(depth_map) else depth_map
        depth_map_np = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())

        if "Inverted" in display_mode:
            color_map = color_map + "_r"

        plt.imsave(output_path, depth_map_np, cmap=color_map)

        gui.update_image(output_path)
        gui.progressbar_1.stop()
    except Exception as e:
        print(f"Error in save_depth_map: {e}")


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


def generate_depth_map(gui, image, model, model_name, log_queue, output_folder, filename, display_mode, color_map,
                       tile_sizes=None):
    try:
        padding = 10  # Define the padding size for each tile
        map_low_location = os.path.join(output_folder, '16bit_low.png')
        map_tile_location = os.path.join(output_folder, "tiles")
        map_done_location = os.path.join(output_folder, "maps", filename)

        if tile_sizes is None:
            tile_sizes = [[2, 2]]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        image_tensor = transforms.ToTensor()(Image.fromarray(image)).unsqueeze(0).to(device)
        log_queue.put(">> Generating low resolution depth map...\n")

        # Generate low resolution depth map
        if "DepthAnythingV2" in model_name:
            raw_img = np.array(image)
            low_res_depth = model.infer_image(raw_img, 518)
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
        low_res_depth_map_image.save(map_low_location)
        log_queue.put(">> Low resolution depth map generated.\n")

        gui.update_image(map_low_location)

        im = np.asarray(image)
        filters = []

        # Generate filters for different tile sizes
        log_queue.put(">> Generating filters for tile sizes.\n")
        for tile_size in tile_sizes:
            num_x, num_y = tile_size
            M, N = im.shape[0] // num_x, im.shape[1] // num_y

            filter_dict = {f'{direction}_filter': np.zeros((M, N)) for direction in
                           ['right', 'left', 'top', 'bottom', 'top_right', 'top_left', 'bottom_right', 'bottom_left']}
            filter_dict['filter'] = np.zeros((M, N))

            for i in range(M):
                for j in range(N):
                    x_value = np.cos((abs(M / 2 - i) / M) * np.pi) ** 2
                    y_value = np.cos((abs(N / 2 - j) / N) * np.pi) ** 2

                    filter_dict['right_filter'][i, j] = x_value if j > N / 2 else x_value * y_value
                    filter_dict['left_filter'][i, j] = x_value if j < N / 2 else x_value * y_value
                    filter_dict['top_filter'][i, j] = y_value if i < M / 2 else x_value * y_value
                    filter_dict['bottom_filter'][i, j] = y_value if i > M / 2 else x_value * y_value

                    filter_dict['top_right_filter'][i, j] = (1 if (j > N / 2 and i < M / 2) else
                                                             x_value if j > N / 2 else
                                                             y_value if i < M / 2 else x_value * y_value)

                    filter_dict['top_left_filter'][i, j] = (1 if (j < N / 2 and i < M / 2) else
                                                            x_value if j < N / 2 else
                                                            y_value if i < M / 2 else x_value * y_value)

                    filter_dict['bottom_right_filter'][i, j] = (1 if (j > N / 2 and i > M / 2) else
                                                                x_value if j > N / 2 else
                                                                y_value if i > M / 2 else x_value * y_value)

                    filter_dict['bottom_left_filter'][i, j] = (1 if (j < N / 2 and i > M / 2) else
                                                               x_value if j < N / 2 else
                                                               y_value if i > M / 2 else x_value * y_value)

                    filter_dict['filter'][i, j] = x_value * y_value

            filters.append(filter_dict)

        log_queue.put(">> Filters for tile sizes generated.\n")

        compiled_tiles_list = []

        # Process tiles
        log_queue.put(">> Starting tile processing.\n")
        for i, tile_size in enumerate(tile_sizes):
            num_x, num_y = tile_size
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

                        # Add padding to the tile
                        padded_tile = cv2.copyMakeBorder(tile, padding, padding, padding, padding, cv2.BORDER_REFLECT)

                        if padded_tile.ndim == 2:
                            padded_tile = np.stack([padded_tile] * 3, axis=-1)
                        elif padded_tile.shape[2] == 1:
                            padded_tile = np.concatenate([padded_tile] * 3, axis=-1)

                        if "DepthAnythingV2" in model_name:
                            depth = model.infer_image(padded_tile, 518)
                        elif "Zoe" in model_name:
                            depth = model(transforms.ToTensor()(Image.fromarray(np.uint8(padded_tile))).unsqueeze(0).to(device))['metric_depth'].squeeze().cpu().detach().numpy()
                        else:
                            depth = model(transforms.ToTensor()(Image.fromarray(np.uint8(padded_tile))).unsqueeze(0).to(device)).squeeze().cpu().detach().numpy()

                        # Remove padding from the depth map
                        depth = depth[padding:-padding, padding:-padding]

                        scaled_depth = 2 ** 16 - (depth - np.min(depth)) * 2 ** 16 / (np.max(depth) - np.min(depth))

                        low_res_slice = low_res_scaled_depth[x:x_end, y:y_end]

                        selected_filter = get_selected_filter(i, x, y, x_coords_all, y_coords_all, filters)

                        if selected_filter.shape == scaled_depth[:selected_filter.shape[0], :selected_filter.shape[1]].shape == low_res_slice.shape:
                            compiled_tiles[x:x_end, y:y_end] += selected_filter * (
                                    np.mean(low_res_slice) + np.std(low_res_slice) * ((scaled_depth[:selected_filter.shape[0], :selected_filter.shape[1]] - np.mean(scaled_depth[:selected_filter.shape[0], :selected_filter.shape[1]])) / np.std(scaled_depth[:selected_filter.shape[0], :selected_filter.shape[1]])))

                        log_queue.put(
                            f">>> Processed tile ({x}, {y}) to ({x_end}, {y_end}) for tile size {tile_size}.\n")
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
            tiled_depth_map.save(os.path.join(map_tile_location, f'tiled_depth_{tile_size}.png'))
            tiled_depth_map_location = os.path.join(map_tile_location, f'tiled_depth_{tile_size}.png')
            log_queue.put(f">>> Tiled depth map for tile size {tile_size} saved.\n")

            gui.update_image(tiled_depth_map_location)

        min_shape = np.min([compiled_tiles_list[0].shape, low_res_scaled_depth.shape], axis=0)
        if len(compiled_tiles_list) > 1:
            min_shape = np.min([compiled_tiles_list[0].shape, compiled_tiles_list[1].shape, low_res_scaled_depth.shape], axis=0)
            combined_result = (compiled_tiles_list[1][:min_shape[0], :min_shape[1]] + compiled_tiles_list[0][:min_shape[0], :min_shape[1]] + low_res_scaled_depth[:min_shape[0], :min_shape[1]]) / 3
        else:
            combined_result = (compiled_tiles_list[0][:min_shape[0], :min_shape[1]] + low_res_scaled_depth[:min_shape[0], :min_shape[1]]) / 2
        combined_result = np.nan_to_num(combined_result)

        # Normalize the combined result
        combined_result = cv2.normalize(combined_result, None, 0, 65535, cv2.NORM_MINMAX)
        combined_result = combined_result.astype("uint16")

        save_depth_map(gui, combined_result, map_done_location, display_mode, color_map)
        log_queue.put(">> High quality depth map generation complete.\n")

        return
    except Exception as e:
        log_queue.put(f"Error in generate_high_quality_depth_map: {str(e)}\n")
        log_queue.put(traceback.format_exc() + "\n")

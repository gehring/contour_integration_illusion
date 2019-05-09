import enum
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

class ContrastLandscape(enum.Flag):
    FIXED = 0
    RANDOM_PATH = enum.auto()
    RANDOM_BACKGROUND = enum.auto()
    SHARED_RANDOM = enum.auto()
    
    
def gabor_kernel(size, scale, wavelength, phase, orientation):
    scale, wavelength, phase, orientation = np.atleast_2d(
        scale, wavelength, phase, orientation)
    
    scale = scale.T
    wavelength = wavelength.T
    phase = phase.T
    orientation = orientation.T
    
    x, y = np.meshgrid(np.linspace(-size[0]/2, size[0]/2, size[0], endpoint=True),
                       np.linspace(-size[1]/2, size[1]/2, size[1], endpoint=True))
    
    # flatten with extra leading dimension to allow broadcasting
    x = x.reshape((1, -1))
    y = y.reshape((1, -1))
    
    # rotate coordinates to match the orientation
    x_r = x*np.cos(orientation) + y*np.sin(orientation)
    y_r = -x*np.sin(orientation) + y*np.cos(orientation)
    
    # compute kernel value at each pixel position
    g = np.exp(-(x_r**2 + y_r**2)/(2*scale**2))
    g = g*np.cos(x_r*2*np.pi/wavelength + phase)
    
    return g.reshape((-1, size[0], size[1])).squeeze()


def add_gabors(positions, gabors, image, clip_values=False):
    image = image.copy()
    min_val = min(image.min(), gabors.min())
    max_val = max(image.max(), gabors.max())
    
    for (x, y), gabor in zip(positions, gabors):
        i = x - gabor.shape[0]//2
        j = y - gabor.shape[1]//2
        
        img_i_start = max(0, i)
        img_j_start = max(0, j)
        img_i_end = min(i + gabor.shape[0], image.shape[0])
        img_j_end = min(j + gabor.shape[1], image.shape[1])
        
        g_i_start = max(0, -i)
        g_j_start = max(0, -j)
        g_i_end = g_i_start + (img_i_end - img_i_start)
        g_j_end = g_j_start + (img_j_end - img_j_start)
        
        image[img_i_start:img_i_end, img_j_start:img_j_end] += gabor[g_i_start:g_i_end,
                                                                     g_j_start:g_j_end]
    if clip_values:
        image = np.clip(image, min_val, max_val)
    return image


def random_grid_positions(grid_size, cell_size):
    img_size = grid_size*cell_size
    offsets = np.stack((np.random.randint(cell_size[0], size=(grid_size[0], grid_size[1])),
                        np.random.randint(cell_size[1], size=(grid_size[0], grid_size[1]))))
    idx = np.mgrid[0:img_size[0]:cell_size[0],
                   0:img_size[1]:cell_size[1]]
    positions = idx + offsets
    return positions.reshape((2, -1)).T
    
    
def grid_indices(point, cell_size):
    return tuple((point/cell_size).astype(np.int64))


def sample_path(path_start, num_points, step_size, grid_size, cell_size,
                path_angles, angle_noise):
    point = np.array(path_start)
    angle = np.arctan2(*(grid_size*cell_size/2 - point))
    grid_occupancy = set()
    
    path = [point]
    for i in range(num_points):
        angle = angle + np.random.choice(path_angles)
        angle = angle + np.random.uniform(-angle_noise, angle_noise)
        angle = angle % (2*np.pi)
        
        direction = np.array((np.sin(angle), np.cos(angle)))
        new_point = point + step_size*direction
        
        element = (point + new_point)/2
        elem_idx = grid_indices(element, cell_size)

        if elem_idx in grid_occupancy:
            new_point = new_point + step_size*direction/4
            element = (point + new_point)/2
            elem_idx = grid_indices(element, cell_size)
        
        # reject path if invalid
        if (elem_idx in grid_occupancy
            or np.any(np.array(elem_idx) < 0)
            or np.any(np.array(elem_idx) >= grid_size)):
                return None
            
        point = new_point
        path.append(new_point)
        grid_occupancy.add(elem_idx)
    return path


def create_path(path_start, num_points, step_size, grid_size, cell_size,
                path_angles, angle_noise, max_tries=1000):
    path = None
    for _ in range(max_tries):
        path = sample_path(path_start, num_points, step_size, grid_size,
                           cell_size, path_angles, angle_noise)
        if path is not None:
            break
        
    return path


def align_position_to_phase(position, wavelength, phase, orientation):
    direction = np.array([np.sin(orientation), np.cos(orientation)])

    phase_shift = wavelength*(phase - np.pi)/(2*np.pi)
    position = position + direction*phase_shift
    return position
    

def create_path_gabor(path, cell_size, size, scale, wavelength,
                      phase=None, align_phase=False):
    positions= []
    gabors = []
    elem_indices = []
    for p1, p2 in zip(path[:-1], path[1:]):
        orientation = np.arctan2(*(p2 - p1)) + np.pi/2
        position = (p1 + p2)/2
        elem_indices.append(grid_indices(position, cell_size))
        
        gabor_phase = phase
        
        # sample a random phase even if not needed to keep the pseudo random
        # generator state consistent.
        rnd_phase = np.random.uniform(0, np.pi*2)
        
        if phase is None:
            gabor_phase = rnd_phase
            
        if align_phase:
            # align the position as though we were using the random phase but use the
            # fixed phase with the new position
            position = align_position_to_phase(position, wavelength, rnd_phase, orientation)
            
        gabor = gabor_kernel(
            size=size,
            scale=scale,
            wavelength=wavelength,
            phase=gabor_phase,
            orientation=orientation,
        )
        
        positions.append(position)
        gabors.append(gabor)
        
    return (np.array(positions, dtype=np.int),
            np.array(gabors),
            elem_indices)


def replace_background_gabor(bg_pos, bg_gabors, path_pos, path_gabors,
                             elem_indices, grid_size, cell_size):
    
    bg_pos = bg_pos.copy()
    bg_gabors = bg_gabors.copy()
    for pos, gabor, grid_idx in zip(path_pos, path_gabors, elem_indices):
        gabor_idx = np.ravel_multi_index(grid_idx, grid_size)
        
        bg_pos[gabor_idx] = pos
        bg_gabors[gabor_idx] = gabor
    return bg_pos, bg_gabors


def uniform_random_contrast(point_grid_size, img_size, min_contrast, max_contrast,
                            epsilon, smooth):
    
    img_size = np.array(img_size, dtype=np.int)
    z = np.random.uniform(min_contrast, max_contrast, size=point_grid_size)
    
    x, y = np.meshgrid(np.linspace(-1.2, 1.2, point_grid_size[0], endpoint=True),
                       np.linspace(-1.2, 1.2, point_grid_size[1], endpoint=True))
    rbf = interpolate.Rbf(x, y, z, epsilon=epsilon, smooth=smooth)
    
    def contrast_function(pos):
        pos = 2*pos/img_size[None, :] - 1.
        return rbf(pos[:, 0], pos[:, 1])
    
    return contrast_function

    
def generate_images(seed, grid_size, cell_size, kernel_size, scale, wavelength,
                    start_distance, num_points, path_angle, angle_noise,
                    random_phase, align_phase, contrast_landscape, contrast_grid_size,
                    min_contrast, max_contrast, generate_contrast_image,
                    contrast_epsilon=0.4, contrast_smooth=0.):
    if seed is not None:
        np.random.seed(seed)
    
    grid_size = np.array((grid_size, grid_size), dtype=np.int)
    cell_size = np.array((cell_size, cell_size), dtype=np.int)
    
    img_size = grid_size*cell_size
    image = np.zeros(img_size)
    
    start_angle = np.random.uniform(np.pi*2)
    path_start = np.array([np.sin(start_angle), np.cos(start_angle)])
    path_start = path_start*start_distance + img_size/2
    path_angles = np.array([-path_angle, path_angle])
    step_size = cell_size[0]

    positions = random_grid_positions(grid_size, cell_size)
    num_gabor = positions.shape[0]
    orientations = np.random.uniform(0, np.pi, size=num_gabor)
    
    phase = np.random.uniform(0., 2*np.pi, size=num_gabor)
    gabors = gabor_kernel(
        size=(kernel_size, kernel_size),
        scale=scale,
        wavelength=wavelength,
        phase=phase if random_phase else np.pi,
        orientation=orientations,
    )

#     bg_image = add_gabors(positions, gabors, image)

    if num_points:
        path = create_path(
            path_start=path_start,
            num_points=num_points,
            step_size=step_size,
            grid_size=grid_size,
            cell_size=cell_size,
            path_angles=path_angles,
            angle_noise=angle_noise,
        )

        path_pos, path_gabors, elem_indices = create_path_gabor(
            path=path,
            cell_size=cell_size,
            size=(kernel_size, kernel_size),
            scale=scale,
            wavelength=wavelength,
            phase=None if random_phase else np.pi,
            align_phase=align_phase,
        )
    
    path_contrast_func = uniform_random_contrast(
        point_grid_size=contrast_grid_size,
        img_size=img_size,
        min_contrast=min_contrast,
        max_contrast=max_contrast,
        epsilon=contrast_epsilon,
        smooth=contrast_smooth,
    )
    
    bg_contrast_func = uniform_random_contrast(
        point_grid_size=contrast_grid_size,
        img_size=img_size,
        min_contrast=min_contrast,
        max_contrast=max_contrast,
        epsilon=contrast_epsilon,
        smooth=contrast_smooth,
    )
    
    bg_contrast = max_contrast
    path_contrast = max_contrast
    if bool(contrast_landscape & ContrastLandscape.SHARED_RANDOM):
        path_contrast_func = bg_contrast_func

    if bool(contrast_landscape & ContrastLandscape.RANDOM_BACKGROUND):
        bg_contrast = bg_contrast_func(positions)[:, None, None]

    if bool(contrast_landscape & ContrastLandscape.RANDOM_PATH):
        path_contrast = path_contrast_func(path_pos)[:, None, None]
        
    gabors *= bg_contrast
    
    path_image = None
    if num_points:
        path_gabors *= path_contrast
        path_image = add_gabors(np.array(path_pos, dtype=np.int),
                            np.array(path_gabors), image)
        positions, gabors = replace_background_gabor(
            positions, gabors, path_pos, path_gabors, elem_indices, grid_size, cell_size)
    
    bg_path_image = add_gabors(positions, gabors, image)
    
    if generate_contrast_image:
        contr_size = (min(img_size[0], 400), min(img_size[1], 400))
        x, y = np.mgrid[0:contr_size[0], 0:contr_size[1]]
        pos = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        
        if bool(contrast_landscape & ContrastLandscape.RANDOM_PATH):
            path_contrast = path_contrast_func(pos).reshape(contr_size)
        else:
            path_contrast = np.broadcast_to(path_contrast, contr_size)
            
        if bool(contrast_landscape & ContrastLandscape.RANDOM_BACKGROUND):
            bg_contrast = bg_contrast_func(pos).reshape(contr_size)
        else:
            bg_contrast = np.broadcast_to(bg_contrast, contr_size)
        
    return path_image, bg_path_image, path_contrast, bg_contrast
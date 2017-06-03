import numpy as np
import cv2

# Identify pixels above the threshold

def color_thresh(img, rgb_thresh=(150, 150, 150), above = True):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    if above:
        above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                    & (img[:,:,1] > rgb_thresh[1]) \
                    & (img[:,:,2] > rgb_thresh[2])
    else:
        above_thresh = ((img[:,:,0] < rgb_thresh[0]) \
                    | (img[:,:,1] < rgb_thresh[1]) \
                    | (img[:,:,2] < rgb_thresh[2])) \
                    & ((img[:,:,0] != 0) | (img[:,:,1] != 0) | (img[:,:,2] != 0)) #not black
    
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Special threshed function for the rock. Uses HSV
def rock_thresh(img):
    
    #color of gold in rgb and HSV 
    gold_rgb = np.uint8([[[220, 180, 30]]])
    gold_hsv = cv2.cvtColor(gold_rgb,cv2.COLOR_RGB2HSV)
 
    # turn image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # create lower and upper limits for gold color
    lower_gold = gold_hsv - np.array([60,80,80])
    upper_gold = gold_hsv + np.array([60,80,80])

    #do the threshold
    rock_threshed = cv2.inRange(img_hsv, lower_gold, upper_gold)

    return rock_threshed

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

# apply_mask chooses which parts of the image to take into account and which to ignore
# For terrain it will choose a circle at the bottom of the screen and a rectangle from the bottom center(see writeup.md)
# For obstacles its a lower rectangle (see writeup.md)
def apply_mask(threshed_img, mask_type):
    
    #Some variables to play with

    #for terrain
    terrain_radius = 40             #the radius of the circle in the bottom
    terrain_corr_width = 15         #the width of the navigable corridor
    terrain_corr_height = 100       #the height of the navigable corridor
    
#    obstacle_radius_scale = 2       #the scale between the bottom rectangle and the radius above 
    obstacle_radius_scale = 1.5       #the scale between the bottom rectangle and the radius above 
 
    obstacle_corr_width_scale = 1   #the scale between the corridor width of the terrain and the obstacle
    obstacle_bottom_height = terrain_radius * obstacle_radius_scale #the height of the bottom rectangle
    obstacle_corr_width = terrain_corr_width * obstacle_corr_width_scale  #the width of the corridor

    #get the dimensions of the thresed image
    nrows, ncols = threshed_img.shape
    row, col = np.ogrid[:nrows,:ncols]
    cnt_row, cnt_col = nrows / 2, ncols / 2

    # Creats an mask of the pixels we want to null
    if mask_type is 'terrain':
        mask = ((nrows-row)**2 + (col - cnt_col)**2 > terrain_radius**2) \
                & ((np.absolute(row - nrows) > terrain_corr_height) \
                | (np.absolute(col - cnt_col) > terrain_corr_width))  
    elif mask_type is 'obstacle':
        mask = (np.absolute(nrows-row) + 0*col > obstacle_bottom_height) #\
 #               | (np.absolute(col - cnt_col) + 0*row < obstacle_corr_width)
    
    # zeroing all the cells we want to ignore
    threshed_img[mask] = 0

    return threshed_img

# This function overlays the steer direction of the rover to the Rover.vision_image for display
def add_travel_direction(Rover):
    #print("SHAPE!!!!",Rover.vision_image.shape)

    arrow_length = 50

    mean_dir = np.mean(Rover.nav_angles)
    if not np.isnan(mean_dir):
        x1 = int(Rover.vision_image.shape[1]/2)
        y1 = Rover.vision_image.shape[0]

        x2 = x1 - int(arrow_length * np.sin(mean_dir))
        y2 = y1 - int(arrow_length * np.cos(mean_dir)) 

        pt1 = (x1,y1)
        pt2 = (x2,y2)

        cv2.arrowedLine(Rover.vision_image, pt1, pt2, (255,0,255), 3)

    return
     


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()

    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img
# 1) Define source and destination points for perspective transform
    
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])
    

# 2) Apply perspective transform
    
    warped = perspect_transform(image, source, destination)    

# 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    # find navigable terrain
    terrain_threshed = color_thresh(warped,rgb_thresh=(140, 140, 140))
    # apply a mask to exclude known noisy areas
    terrain_threshed_masked = apply_mask(terrain_threshed, mask_type = 'terrain')
    
    # find obstacles
    obstacle_threshed = color_thresh(warped, rgb_thresh=(100, 100, 100),  above=False)
    # apply a mask to exclude known noisy areas
    obstacle_threshed_masked = apply_mask(obstacle_threshed, mask_type = 'obstacle')

    # find rocks
    rock_threshed = rock_thresh(warped)
    

# 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image


    Rover.vision_image[:,:,0] = obstacle_threshed_masked*255
    Rover.vision_image[:,:,1] = rock_threshed
    Rover.vision_image[:,:,2] = terrain_threshed_masked*255
    

# 5) Convert map image pixel values to rover-centric coords

    terr_xpix, terr_ypix = rover_coords(terrain_threshed_masked)
    ob_xpix, ob_ypix = rover_coords(obstacle_threshed_masked)
    rock_xpix, rock_ypix = rover_coords(rock_threshed)

# 6) Convert rover-centric pixel values to world coordinates

    world_size = Rover.worldmap.shape[0]
    scale = 10
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw

    terr_x_world, terr_y_world = pix_to_world(terr_xpix, terr_ypix, xpos, ypos, yaw, world_size, scale)
    obstacle_x_world, obstacle_y_world = pix_to_world(ob_xpix, ob_ypix, xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)


# 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    #add points only if the camera is stable
    stable_cam = ((Rover.pitch > 359) | (Rover.pitch < 1)) \
                & ((Rover.roll > 359) | (Rover.roll < 1))
    print("STABLE_CAM =", stable_cam)
    
    if (stable_cam):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 2] -= 1
        Rover.worldmap[terr_y_world, terr_x_world, 2] += 3

    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    
# 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    if (len(rock_xpix) != 0):
        print("GOING AFTER GOLD")
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(rock_xpix, rock_ypix)
        Rover.mode = 'get_sample'
    else:
        #using unmasked terrain to get full vision line of sights
        terr_xpix_all, terr_ypix_all = rover_coords(terrain_threshed)
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(terr_xpix_all, terr_ypix_all)  

        #using unmasked terrain to get full vision
#    terr_xpix_all, terr_ypix_all = rover_coords(terrain_threshed)
#    Rover.nav_dists, Rover.nav_angles = to_polar_coords(terr_xpix_all, terr_ypix_all)  


    #add arrow showing direction of travel to the displayed image in Rover.vision_image
    add_travel_direction(Rover)
    
    return Rover
## Project: Search and Sample Return


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  


## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

## Notebook Analysis
### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

A large part of the notebook is pre-written code. I will only address the part of the code which I changed.

#### ___Color thresholding___

[thresh_image1]: ./misc/writeup/jupyter/perspective.png
[thresh_image2]: ./misc/writeup/jupyter/threshed_terr.png
[thresh_image3]: ./misc/writeup/jupyter/threshed_obst.png
[thresh_image4]: ./misc/writeup/jupyter/rock.png
[thresh_image5]: ./misc/writeup/jupyter/threshed_rock.png 

In order to threshold the navigable terrain I used the code that was given in the lecture. This analyzed pixel values that where under a certain threshold. I changed it a bit so I could use the same function for finding obstacles. For obstacles I looked for that where below a certain threshold. So I added a above flag to the function.

```python
def color_thresh(img, rgb_thresh=(160, 160, 160), above = True):
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
```

Original warped image:

![alt text][thresh_image1]

Warped image after threshold. On the left for navigable terrain and on the right for obstacles 

![alt text][thresh_image2]
![alt text][thresh_image3]


For the rock sample I created another function that uses HSV color space. I did this because it is more robust to brightness and I was having problems finding thresholds in the RGB color space. Here is the code for the special function:

```python
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
```
&nbsp;

Here are the result of the threshold on a camera image:

![alt text][thresh_image4]
![alt text][thresh_image5]

### ***Coordinate Transformations***

Here I used the functions that where shown in the lessons.

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

[process_image1]: ./misc/writeup/jupyter/terrain_mask.png
[process_image2]: ./misc/writeup/jupyter/obstacle_mask.png
[process_image3]: ./misc/writeup/jupyter/video_screenshot.png


In order to keep `process_image()` "clean" I added a cell above with the `apply_mask()` function which I'll explain further down this document. I will now go over different parts of the function:

1) The perspective calibration calculations where given. I didn't change this part of the code
2) For the warped part of the code I created a warped image using the `perspective_transform()` function.
3) In order to identify navigable terrain/obstacles/rock samples I improved on the original algorithm. The first thing I did was to apply `color_thresh()` and the `rock_thresh()` on the warped image in order to create the threshold images for:

* navigable terrain
* obstacles
* rock

I saw that the the threshold for the warped images for the navigable terrain and obstacle introduced large errors certain areas of the images. For the navigable terrain this was far away points, especially on the sides. For the obstacles this was far away points. That is why I passed these images through a filter where I ignored these problematic points. I created masks in order to identify these regions. This is the `apply_mask()` function.

The following images depict the the area which were ignored. The red regions are ignored the green regions are kept. Black region are the blind spots of the camera

#### __Navigable terrain:__

![alt text][process_image1]

#### __Obstacles__

![alt text][process_image2]

Here is the code:

```python
def apply_mask(threshed_img, mask_type):
    
    #Some variables to play with

    #for terrain
    terrain_radius = 40             #the radius of the circle in the bottom
    terrain_corr_width = 15         #the width of the navigable corridor
    terrain_corr_height = 100       #the height of the navigable corridor
    
    obstacle_radius_scale = 1.5       #the scale between the bottom rectangle and the radius above 
 
    obstacle_corr_width_scale = 1   #the scale between the corridor width of the terrain and the obstacle
    obstacle_bottom_height = terrain_radius * obstacle_radius_scale #the height of the 

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
        mask = (np.absolute(nrows-row) + 0*col > obstacle_bottom_height)
    elif mask_type is 'display_line':
        mask = (np.absolute(nrows-row) + 0*col < obstacle_bottom_height) \
                  | (np.absolute(nrows-row) + 0*col > obstacle_bottom_height + 5)

    # zeroing all the cells we want to ignore
    threshed_img[mask] = 0

    return threshed_img
```

For convenience and for debugging purposes, I added the distance I used in  `obstacle_threshold_mask` on top of the warped image, which will displayed in the video. I marked this distance as a red line. This is the 'display_line' part of the code.

4) For the rover-centric coordinates I just used the correct function

5) For the rover-centric coordinate to the world map, again I used the correct functions

6) In order to update the world map I added 10 to the red channel of `Rover.worldmap` for every pixel in the world map that was identified as a obstacle pixel. I added 15 to the blue channel of `Rover.worldmap` for every pixel that was identified as navigable terrain. This implements a weighted map that gives preference to the navigable terrain.  

7) For the display I used the same code that was given but made a few modifications. 

* I made sure to clip the `Rover.worldmap` variables before I overlaid them on `data.groundtruth` so there wasn't any overflow.
* I added an image that shows the thresholds of the navigable terrain in blue, obstacle in red and rock sample in green
* I wrote the name of the file that is being processed on top

Here is a screenshot:

![alt text][process_image3]

Note: The video can be found in the output folder.

## Autonomous Navigation and Mapping

### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

[autonomous_image1]: ./misc/writeup/debug_arrow.png
[autonomous_image2]: ./misc/writeup/stuck.png

At any given time the robot was in one of three states: 

1. 'forward' mode
2. 'stop' mode
3. 'get_sample' mode

I will elaborate in more detail in the `decision_step()` part of the code

#### Perception Step 

Most of the `perception_step()` code is the same as the in the Jupyter notebook. For the image take by the camera the image is:

1. Warped using `perspect_transform()`
2. Do color thresholding for the navigable terrain, obstacles and rock samples using `color_thresh()` and `rock_thresh()`.
3. Create masked thresholds for the navigable terrain and obstacles using the `apply_mask()` function. 
_Note: These will be used for the mapping, and not for the steering._
4. Turn the masked thresholds (navigable terrain and obstacles) and the rock sample thresholds into rover centric coordinates using the `rover_coords()` function
5. Turn these into world map coordinates using the `pix_to_world()` function
6. Check to see if the camera is stable. This means that the pitch and the roll are no more than 1 degree. If so it is safe to add the points to `Rover.worldmap` otherwise they are too noisy. 
_Note: This helps allot with fidelity._
7. If the camera is unstable don't update `Rover.worldmap` with the values from the navigable terrain and obstacles. Otherwise update `Rover.worldmap`. This is done a little different than was implemented in the notebook:

	a. In the red channel of `Rover.worldmap` add `1` to all the pixels that had detected an obstacle
	
    b. In the blue channel of `Rover.worldmap` add `3` to all the pixels that had detected navigable terrain. Subtract `1` from all the pixels that had detected obstacles. 

	_Note: The weighted sum on the blue channel helps allot with fidelity_
8. Even if the camera is unstable, update the green channel of `Rover.worldmap` by adding one to all the pixels that had detected a rock sample.
9. Check to see if the robot has detected rock sample pixels in the robot central coordinate system?
	* If it has convert these point to polar coordinates. Then push these values to `Rover.nav_dists` and `Rover.nav_angles` (which will be used for steering) and change the robot state to **'get_sample'** mode.
	* If not than get the rover centric coordinates of the whole threshold image of whole line of sight (not just the masked part of the image which was used for updating the map). These points are found by running `rover_coords(terrain_threshold)` and stored in appropriate x and y variables. These points are then turned into polar coordinates and are pushed into the `Rover.nav_dists` and `Rover.nav_angles` variables (again to be used for steering). The robot's state is unchanged.
10. `add_travel_direction()` function is used to add an arrow whose direction is the average of `Rover.nav_angles` in the simulator on top of image that displays the different thresholds. This is useful for debugging. 

	![alt text][autonomous_image1]

&nbsp;

#### Decision Step

As was said earlier, at any given moment the robot is in one of three states. These are implemented in the `decision_step()`. The algorithm does the following. First it checks to see if it was given steering angles in `Rover.nav_angles`. If it wasn't given steering angles than the robot drives forward. If it was than the robot check to see in which state it is in.

If its in _'forward'_ state then it checks to see there is enough navigable terrain in front of the robot. If there isn't than the robot is stopped (throttle set to zero and brakes on) and the robot changes state to _'stop'_ state. If there is enough navigable terrain than the robot checks to see if its stuck (the throttle is on but not moving). If this is the case than the robot tries to get unstuck. If it isn't than the robot checks to see if its in an open area or a narrow corridor in order to decide if to use wall crawling. Wall crawling is implemented when the robot reaches large open areas. The steering angle is calculated by averaging the `Rover.nav_angles`. To implement wall crawling the standard deviation (times a certain scaling factor) is subtracted from the steering angle. This makes the robot stick to the right wall. The steering angles are then clipped and if the robot is moving really fast but wants to perform a hard turn than a bit of a brake is applied.

If the robot is in _'stop'_ state and the robot is still moving the brake is turned on and the steering is in the direction of the average and of `Rover.nav_angles`. If it's not moving it checks to see if there is enough navigable terrain in front of the robot. If there is than the robot start to move in the correct direction and changes the robot state to _'forward'_. If there isn't than the robot does an 'on the spot' turn (and stays in the _'stop'_ state). 
    
If the robot is _'get_sample'_ state than we check the `Rover.near_sample` flag. If it is true than we stop the robot. If it isn't than the robot turns in the direction on the rock sample. The robot is then changed to the _'forward'_. _Note: There is a good chance that the robot will stay in the 'get_sample' state and not the 'forward' state since the state can still change in `perception_step()` before the next cycle of `decision_step()`_

The last part of the `decision_step()` is to check to see if the robot isn't moving and is near the rock sample. If it is than the robot picks up the sample.


### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

The setting I used for running the simulator are as follows:

#### Screen resolution: 1680x1050
#### Graphics quality: Good
#### Frames per second: 30 fps

There where a few areas where the robot acted well and a few areas that could have been improved. One thing that worked well was the mapping. I was able to keep the fidelity above 80%. The things that contributed to this where the masking, ignoring the points when the pitch or roll where too high and subtracting the obstacle pixels from the navigable terrain pixels channel in `Rover.worldmap`. 

The rock picking worked well and was able to pick up rock sample in most cases. It had a hard time if the rock samples where placed in difficult areas such as behind certain rocks. If the rock is very far and the robot is moving fast it would sometimes miss the rock but would pick up the rock when it revisited the region. 

The wall crawling worked well and help the robot reach the whole map and find all the rock samples. Without wall crawling there where cases where the robot will only visit certain channels (corridors) and not visit the whole map.

The thing that didn't work too well algorithm for getting unstuck. It worked in some cases where the robot could wiggle it's way out of certain situations and didn't work in other situations. There is also a certain area of the map where the robot always got stuck. You can see it in the image below. In this situation the robot sees navigable terrain and tries to drive forward even though it can't. In this state the only way to get the robot unstuck it to manually move it. 


![alt text][autonomous_image2]

There are also cases where the robot can wiggle it's way unstuck but it takes a long time. 

In certain start angles (when the simulator if fired up) the robot also has a problem. The wall creeping makes the robot go right and since the starting point is a large open area than the robot goes around in a circles and has a hard time finding a channel. At a certain point the robot hits rocks which makes it change direction and is able to find a channel an explore, but this takes time and could probably be improved.

Things I would have worked on to improve the robot would have been:

1. Improve the unstuck algorithm. Maybe in certain cases turn 180 degrees in place. I could also maybe use the roll and the pitch coupled with the robot speed to identify collisions.
2. I could played with speed accelerations and brakes so that the robot collides less with obstacles. 
3. Understand why in certain cases the robot doesn't visit certain certain areas of the map on it first go.
4. Add a return to home state
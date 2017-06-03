import numpy as np

def get_unstuck(Rover):
    Rover.throttle = 0
    # Release the brake to allow turning
    Rover.brake = 0
    # Get the steering angle direction
    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
    
    angle = np.mean(Rover.nav_angles * 180/np.pi)
    if Rover.steer != 0:

        if (abs(angle) < 10):
            Rover.steer = 15
        else:
            Rover.steer = np.sign(angle) * 15
    else:
        Rover.steer = 15
    return
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        #print the mode
        print("MODE: ",Rover.mode, "!!!!!!!!!!!!!!!!!")
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                #try to get unstuck
                if (Rover.throttle == Rover.throttle_set) and (Rover.vel <= 0):
                    get_unstuck(Rover)
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                else:
                    if Rover.vel < Rover.max_vel:
                        # Set throttle value to throttle setting
                        Rover.throttle = Rover.throttle_set
                    else: # Else coast
                        Rover.throttle = 0
                    Rover.brake = 0
                    # Set steering to average angle clipped to the range +/- 15
                    aver_angle = np.mean(Rover.nav_angles * 180/np.pi)
                    deviation = np.std(Rover.nav_angles * 180/np.pi)
                    print("Number of nav_angle points=", len(Rover.nav_angles),"###########################")
                    
                    #check if on open plain or corridor
                    if (len(Rover.nav_angles) > 1000):
                        angle = aver_angle - deviation/3
                        print("wall crawling")
                    else:
                        angle = aver_angle
                        print("NOT wall crawling")
                    print("STEERING ANGLE = ", angle)
                    Rover.steer = np.clip(angle, -15, 15)
                    if (((angle > 15) | (angle < -15)) & (Rover.vel > Rover.max_vel/2)):
                        Rover.brake = 10

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
#                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
#                Rover.steer = 0
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    #Rover.steer = -15 # Could be more clever here about which way to turn
                    if Rover.steer != 0:
                        Rover.steer = np.sign(Rover.steer) * 15
                    else:
                        Rover.steer = 15
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
        #get the sample
        elif Rover.mode == 'get_sample':
            #if (np.mean(Rover.nav_dists) < 15):
            if (Rover.near_sample == 1):
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            #if try to get unstuck
            elif (Rover.throttle == Rover.throttle_set) and (Rover.vel <= 0):
                get_unstuck(Rover)
            elif (np.absolute(np.mean(Rover.nav_angles * 180/np.pi)) > 20):
                if Rover.vel > 0.2:
                    Rover.throttle = 0
#                    Rover.brake = 0.4
                    Rover.brake = 5
                    Rover.steer = 0
                # If we're not moving (vel < 0.2) then do something else
                elif Rover.vel <= 0.2:
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = np.sign(np.mean(Rover.nav_angles)) * 15        
            else:
                if Rover.vel < 1.5:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                # Release the brake
                Rover.brake = 0
                # Set steer to mean angle
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)                
            
            # set in case rock isn't detected next time
            Rover.mode = 'forward'            
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print("Something bad happened")
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        Rover.mode = 'forward'
    
    return Rover


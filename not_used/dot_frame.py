# Draws dots at the corner coordinates
import cv2
def draw_corner_dots(image_path, coordinates):
    # Load the image
    image = cv2.imread(image_path)

    # Extract coordinates
    x1, y1, x2, y2 = coordinates

    # Draw the dots on the image
    dot_radius = 5
    dot_color = (0, 0, 255)  # Red color
    thickness = -1  # Fill the dot
    image_with_dots = cv2.circle(image.copy(), (x1, y1), dot_radius, dot_color, thickness)
    image_with_dots = cv2.circle(image_with_dots, (x2, y1), dot_radius, dot_color, thickness)
    image_with_dots = cv2.circle(image_with_dots, (x1, y2), dot_radius, dot_color, thickness)
    image_with_dots = cv2.circle(image_with_dots, (x2, y2), dot_radius, dot_color, thickness)

    # Display the image with the dots
    cv2.imshow("Image with Corner Dots", image_with_dots)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Define the image path and corner coordinates
image_path = "../datasets/cap_frame/frame_0.jpg"
coordinates = [0, 352, 291, 635]

# Call the function
draw_corner_dots(image_path, coordinates)

# Import required packages
import cv2
import socket
import numpy as np
# import pytesseract
def doAI(img):
	# Mention the installed location of Tesseract-OCR in your system
	# pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
	print("Doing CV")
	# Read image from which text needs to be extracted

	# Preprocessing the image starts

	# Convert the image to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Performing OTSU threshold
	ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

	# Specify structure shape and kernel size.x
	# Kernel size increases or decreases the area
	# of the rectangle to be detected.
	# A smaller value like (10, 10) will detect
	# each word instead of a sentence.
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

	# Applying dilation on the threshold image
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

	# Finding contours
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
													cv2.CHAIN_APPROX_NONE)

	# Creating a copy of image
	im2 = img.copy()

	# A text file is created and flushed
	file = open("recognized.txt", "w+")
	file.write("")
	file.close()

	# Looping through the identified contours
	# Then rectangular part is cropped and passed on
	# to pytesseract for extracting text from it
	# Extracted text is then written into the text file
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		
		# Drawing a rectangle on copied image
		rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
		# Cropping the text block for giving input to OCR
		cropped = im2[y:y + h, x:x + w]
		cv2.imwrite("1.jpg", cropped)
		
		# Open the file in append mode
		file = open("recognized.txt", "a")
		
		# Apply OCR on the cropped image
		text = "WORKS"
		
		# Appending the text into file
		file.write(text)
		file.write("\n")
		
		# Close the file
		file.close

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
server_address = ('localhost', 10000)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    print('waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print('connection from', client_address)

        while True:
            # Receive the input data as a byte array
            data = connection.recv(100000)
            if data:
                # Convert the data to a numpy array
                img_array = np.frombuffer(data, dtype=np.uint8)

                # Load the numpy array as an image using cv2
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

                # cv2.imwrite(".\output.jpg", img)

                # Process the image and get the output
                doAI(img)
                # output = [str(x) for x in output]  # Convert each element to a string
                # output = ",".join(output)
                output = ""
                if output == "":
                    output = "unrecognized"
                print("OCR output: " + output)

                # Send the output back to the client
                connection.sendall(output.encode('utf-8'))

            else:
                # No more data, so break out of the loop
                break
    # except:
    #     print("oups")
    finally:
        # Clean up the connection
        print("Closed connection")
        connection.close()

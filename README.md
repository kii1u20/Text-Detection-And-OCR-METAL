# Text-Detection-And-OCR-METAL

Text detection localhost server that receives an array of bytes as input, feeds the array to a tensorflow model for recognising text and then to another tensorflow model for reading that text. The system has been specifically optimised to run on the Apple Metal API for accelerated inference on macOS devices.

To run, simply execute the server.py script, and it will load the server on localhost. Then in your program, create a socket that connects to localhost and send images to that in the format of a byte array.

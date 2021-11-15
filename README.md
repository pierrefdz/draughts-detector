The project was done as a part of a Computer Vision course given by Renaud Keriven. The goal was to apply the methods learned in the lectures. Here, we use feature extraction, clustering methods and some other algorithms to get information from a chessboard picture. The full report is available in the pdf file. 

The code allows to get a simple matrix form from a picture of a chessboard.

The GridDetection file contains the functions used to detect, order and group the lines.
It also contains the scoring methods used to find the good homography.

The PawnsDetection file contains the function used to detect and distinguish pawns in a chessboard.

A set of picture already exists in the folder data. 
You can use other pictures if you want to. To do so, you have to put the picture in the folder data and change the path of the picture in the main.cpp.
It can be of any size and any extension, however the chessboard must be 8x8 in order for the algorithms to work.

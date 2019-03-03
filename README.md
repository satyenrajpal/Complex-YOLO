# Complex-YOLO
#### NOTE: IN PROGESS! 
PyTorch Implementation of [Complex-YOLO](https://arxiv.org/abs/1803.06199) 3D<br>
Much of the code has been adapted from [this repo](https://github.com/AI-liu/Complex-YOLO). However, I've changed a considerable amount of code, improved readibility and speed (from initial runs). <br>
Update:
 - 3/3 - Fixed bug in loss. Commented `region_loss.py` and `utils.py` in relation to notation in paper.
 - 3/2 - Not able understand what loss function was doing. Bug in loss. 

## TODO:
 - [x] Go through Kitti Dataset class definition
 - [x] Construct model from original implementation
 - [x] Download data
 - [x] Modify loss function
 - [x] Comment loss function. Understand everything that's going on
 - [ ] Test function
 - [ ] Visualization of results
 - [x] Running loss for saving model
 - [ ] Streamline code in `region_loss.py`. Two for loops are unnecessary

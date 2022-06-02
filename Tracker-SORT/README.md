
Simple tracker(cot the cookies!) for anyone t get started! 


So lets go! 

Recently a friend of mine reached out to me how can we use a tracker. Here is my one cent, use SORT and they are robust and if you would like, update to CNN SORT.  


A set of trackers use Kalman filter to predict the future state of the box based on its velocity. Next, each is associated with a list of detections by calculating IOU metric and solving the linear assignment problem. Input noise is smoothed by averaging predictions and detections. If a tracker is lost due to occlusion for a prolonged period while the object changes direction, the accumulated momentum may cause it to diverge from its real path.

A short-term object re-identification is implemented. Unmatched trackers are associated with new detections using a collinearity metric. Cost function computes the dot product between vectors of new detection and the last few averaged. This improves the result for sparse crowds with highly non-linear motion, but may also cause some identity swaps in denser areas, as seen on 0:05 where false positive is assigned to id1.2

pip install scikit-image
pip install numba
pip install filterpy
# poisson2D

Solve the 2D poisson equation on any domain filled by a point cloud, using the least-squares approach outlined in
 
*Grid free method for solving the Poisson equation, J. Kuhnert, S. Tiwari (2001)*

* Details: doc/doc.pdf
* Usage: meshlessFish.solvePoisson( ... )
* Solve for first derivatives in a point cloud: meshlessFirstDeriv.getFirstDerivs( ... )
* Solve for 0th derivative (interpolate) in a point cloud: meshlessZerothDeriv.interpolate( ... )


Poisson is French for fish and the poisson equation is named after a French guy. 
So long and thanks for the fish, Mr. Fish!



![example1](example1/example1_result.png?raw=true "Demonstration")

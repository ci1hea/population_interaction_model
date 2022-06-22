# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:18:56 2020

@author: cip18jjp
"""


def count_boxes(data, box_size, range_box ):
    import numpy as np
    
    """

    Parameters
    ----------
    data : Pandas series
        data consisting of x and y coordinates
    box_size : array
        Of box lengths (m).
    range_box : int
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    
    xdata = data['x-coord']
    xmin = xdata.values.min()
    ydata = data['y-coord']
    ymin = ydata.values.min()
    
    N = np.int( np.floor( range_box / box_size ) ) #No full boxes in the range
    
    counts = list()
    
    for i in range( N ):
        for j in range(N):
            xcondition = ( xdata >= xmin + i*box_size )&( xdata < xmin + (i+1)*box_size)
            ycondition = ( ydata >= ymin + j*box_size )&( ydata < ymin + (j+1)*box_size)
            
            subsetx = xdata[ xcondition ].index #return x indices where true
            subsety = ydata[ ycondition ].index #return y indices where true
            
            newid = subsetx.intersection(subsety)
            
            counts.append( xdata[newid].count() )
    
    counts = [ i for i in counts if i != 0 ]
    return len( counts )


def f_temp( x, A, Df ):
    '''
    User defined function for scipy.optimize.curve_fit(),
    which will find optimal values for A and Df.
    '''
    return Df * x + A


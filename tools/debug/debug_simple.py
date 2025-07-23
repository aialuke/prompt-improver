#!/usr/bin/env python3
"""Simple debug to isolate issue"""

import asyncio
import numpy as np

async def test_simple():
    # Test simple numpy operation
    arr = np.array([1, 2, 3])
    print(f"Array: {arr}")
    
    # Test awaiting an async function that returns numpy array
    async def get_array():
        return np.array([4, 5, 6])
    
    result = await get_array()
    print(f"Result: {result}")

asyncio.run(test_simple())
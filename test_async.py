import time
import asyncio
import aiohttp

var = 1

async def random_number():
    time.sleep(10)
    global var
    var = 10

def print_number():
    print(var)
    return var+1
    

async def main():
    a = await random_number()
    c = await random_number()
    b = print_number()
    print(b)
    
asyncio.run(main())
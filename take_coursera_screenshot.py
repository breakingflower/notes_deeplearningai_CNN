import sys
import subprocess
import pyperclip
import pyscreenshot as ImageGrab

if __name__=="__main__": 

    if (len(sys.argv) < 2) or (len(sys.argv)>3): 
        "Expected only a single argument: the desired filename"
        sys.exit(1)

    image_name = sys.argv[1]
    
    # hardcoded bbox for when you have two
    # screens and its in the left screen
    bbox = (364, 322, 1290, 845)

    # take image
    im = ImageGrab.grab(bbox=bbox)
    im.save(f"{image_name}.png")

    # copy the markdown command to clipboard
    pyperclip.copy(f"![{image_name}]({image_name}.png)")

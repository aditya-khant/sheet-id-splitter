import argparse
import glob
import pathlib
import shutil
import sys

import PIL.Image, PIL.ImageTk
from tkinter import *
from tkinter import ttk

# Getting input
parser = argparse.ArgumentParser(description='A tool for labeling images detected as bars as correct or incorrect.')
parser.add_argument('directory', type=str, nargs='?', default='.',
                    help='which directory the unlabeled images are in')
parser.add_argument('--correct_dir', type=str, nargs='?',
                    default='./correct_bars', help='which directory the correct bars are put')
parser.add_argument('--incorrect_dir', type=str, nargs='?',
                    default='./incorrect_bars', help='which directory the incorrect bars are put')

args = parser.parse_args()
print('directory to label: {}\ndirectory to place correct images: {}\ndirectory\
to place incorrect images: {}'.format(args.directory, args.correct_dir,
                                      args.incorrect_dir))

# hack to make sure glob works
args.directory += '/' if args.directory[-1] != '/' else ''
args.correct_dir += '/' if args.correct_dir[-1] != '/' else ''
args.incorrect_dir += '/' if args.incorrect_dir[-1] != '/' else ''

# Make the output directories if they don't exist already
pathlib.Path(args.correct_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(args.incorrect_dir).mkdir(parents=True, exist_ok=True)

filenames = glob.iglob(args.directory + '*.png')

# GUI initialization
root = Tk()
root.title('Bar labeler')
mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Main part of the GUI
description = ttk.Label(mainframe, text='Press "b" to label as a bar, "x" to label as not a bar.')
description.grid(columnspan=2,row=1, sticky=(N,E,W))
# create the image label
image_label = ttk.Label(mainframe, image=None)
image_label.grid(columnspan=2, row=2, sticky=(N, E, W))
# displays the location where the last bar file was placed
image_file = StringVar()
bar_file = ttk.Label(mainframe, textvariable=image_file)
bar_file.grid(columnspan=2, row=4, sticky=(S, E, W))


# GUI functions
def correct_image(*_):
    '''
    Moves the current image to the correct images directory.
    '''
    print('Moving {0} to {1}'.format(image_file.get(), args.correct_dir))
    shutil.move(image_file.get(), args.correct_dir)
    change_image()

def incorrect_image(*_):
    '''
    Moves the current image to the incorrect images directory.
    '''
    print('Moving {0} to {1}'.format(image_file.get(), args.incorrect_dir))
    shutil.move(image_file.get(), args.incorrect_dir)
    change_image()

def change_image():
    '''
    Updates the current image.
    '''
    next_image_file = None
    try:
        next_image_file = next(filenames)
    except StopIteration:
        # We're done if there are no more files
        sys.exit(0)
    print('loading image {}'.format(next_image_file))
    im = PIL.Image.open(next_image_file)
    ph = PIL.ImageTk.PhotoImage(im)
    image_label.configure(image=ph)
    # keep a reference to the photo
    image_label.image = ph
    # keep a reference to the filename
    image_file.set(next_image_file)

# buttons
ttk.Button(mainframe, text='Incorrect', command=incorrect_image).grid(column=0,row=3,sticky=(N))
ttk.Button(mainframe, text='Correct', command=correct_image).grid(column=1,row=3,sticky=(N))

# call the change image function to update the display
change_image()

root.bind('b', correct_image)
root.bind('x', incorrect_image)
root.mainloop()

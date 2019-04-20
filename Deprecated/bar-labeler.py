# misc helpers
import argparse
import glob
import pathlib
import shutil
import sys

# GUI
import PIL.Image, PIL.ImageDraw, PIL.ImageTk
from tkinter import *
from tkinter import ttk

# number of pixels to the left and right of the supposed bar
BAR_CONTEXT = 10
# number of pixels to draw the supposed barline
NUM_PIXELS = 14

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

# hack to make sure the glob works
args.directory += '/' if args.directory[-1] != '/' else ''
args.correct_dir += '/' if args.correct_dir[-1] != '/' else ''
args.incorrect_dir += '/' if args.incorrect_dir[-1] != '/' else ''

# Make the output directories if they don't exist already
pathlib.Path(args.correct_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(args.incorrect_dir).mkdir(parents=True, exist_ok=True)

# TODO: Get this cropping
# # Parse the input files
# def get_bars(staff_image_filename, bar_context = 10):
#     '''
#     Helper function to return an iterator that gives tuples of (staff_image,
#     bar_image), where the former is in a PIL ImageTk format (a PhotoImage) and
#     the latter is in a PIL Image format.

#     Expects that the directory has files that look like
#         staff_image.png -- The image of staff.
#         staff_image.txt -- A file where every line denotes where a bar is in staff_image.png.

#     '''
#     extension_len = len(staff_image_filename.split('.')[-1])
#     bar_indices_filename = staff_image_filename[:-extension_len] + '.txt'
#     with open(bar_indices_filename) as bar_indices_file:
#         bar_indices = (int(line.strip()) for line in bar_indices_file)
#     im = PIL.Image.open(staff_image_filename)
#     im_width, im_height = im.size
#         for bar_index in bar_indices:
#             start_col = max(bar_index - bar_context, 0)
#             end_col = min(bar_index + bar_context, im_width - 1)
#             start_row = 0
#             end_row = im_height - 1
#             bar_drawn = im.copy()
#             draw = PIL.ImageDraw.Draw(bar_drawn)
#             # left border
#             draw.line((start_col, start_row, start_col, end_row), fill=(0, 255, 0), width=3)
#             # centerline
#             draw.line((bar_index, start_row, bar_index, end_row), fill=(255, 0, 0), width=1)
#             # right border
#             draw.line((end_col, start_row, end_col, end_row), fill=(0, 255, 0), width=3)
#             cropped_bar = im.crop(start_col, start_row, end_col, end_row)
#             yield PIL.ImageTk.PhotoImage(bar_drawn), cropped_bar

filenames = glob.iglob(args.directory + '*.png')

# GUI initialization
# Pretty much taken entirely from the ttk docs
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
image_label.grid(columnspan=2, row=2, sticky=(N, E, W, S))
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
    # Open and convert to RGBA
    im = PIL.Image.open(next_image_file).convert('RGB')
    im_width, im_height= im.size
    drawer = PIL.ImageDraw.Draw(im, mode='RGB')
    # draw the centerline at the top
    half_width = im_width // 2
    half_len   = NUM_PIXELS // 2
    # top triangle
    drawer.polygon([(half_width - half_len, im_height - 1), (half_width + half_len, im_height - 1), (half_width, im_height - (1 + NUM_PIXELS))], fill='red')
    drawer.polygon([(half_width - half_len, 0), (half_width + half_len, 0), (half_width, NUM_PIXELS - 1)], fill='red')
    del drawer
    # drawer.line((im_width // 2, im_height - 1, im_width // 2, im_height - (1 + NUM_PIXELS)), fill='red', width=1)
    # draw the centerline at the bottom
    # drawer.line((im_width // 2, NUM_PIXELS - 1, im_width // 2, 0), fill='red', width=1)
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
# start the GUI
root.mainloop()

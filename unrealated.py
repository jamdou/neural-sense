import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import numpy as np

def sort_colours():
  colours = []
  for colour in mc.CSS4_COLORS:
    colours.append(colour)

  colours_lum = np.array([0.2126*mc.to_rgb(colour)[0] + 0.7152*mc.to_rgb(colour)[1] + 0.0722*mc.to_rgb(colour)[2] for colour in colours])
  order = np.argsort(colours_lum)
  
  # sorted_colours = []
  # for colour_index in range(len(colours)):
  #   for order_index in range(len(order)):
  #     if order[order_index] == colour_index:
  #       sorted_colours.append(colours[order_index])

  print(colours_lum)
  sorted_colours = []
  sorted_categories = []
  duplicated_colours = []
  duplicated_categories = []
  sorted_label_index = 0#97
  has_started = False
  has_started_temp = False
  for order_index_index, order_index in enumerate(order):
    if has_started:
      if np.round(64*colours_lum[order[order_index_index - 1]]) < np.round(64*colours_lum[order_index]):
        sorted_label_index += 1
        has_started_temp = False
      else:
        if not has_started_temp:
          duplicated_colours.append(colours[order[order_index_index - 1]])
          duplicated_categories.append(sorted_label_index)
          has_started_temp = True
        duplicated_colours.append(colours[order_index])
        duplicated_categories.append(sorted_label_index)
        
    else:
      has_started = True
    sorted_colours.append(colours[order_index])
    sorted_categories.append(sorted_label_index)

  emptycols=0
  cell_width = 212
  cell_height = 22
  swatch_width = 48
  margin = 12
  topmargin = 40

  n = len(sorted_colours)
  ncols = 4 - emptycols
  nrows = n // ncols + int(n % ncols > 0)

  width = cell_width * 4 + 2 * margin
  height = cell_height * nrows + margin + topmargin
  dpi = 72


  fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
  fig.subplots_adjust(margin/width, margin/height,
                      (width-margin)/width, (height-topmargin)/height)
  ax.set_xlim(0, cell_width * 4)
  ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)
  ax.set_axis_off()
  ax.set_title("Colours by luminance", fontsize=24, loc="left", pad=10)

  for i, zipp in enumerate(zip(sorted_colours, sorted_categories)):
    name, cat = zipp
    row = i % nrows
    col = i // nrows
    y = row * cell_height

    swatch_start_x = cell_width * col
    text_pos_x = cell_width * col + swatch_width + 7

    ax.text(text_pos_x, y, f"{cat} - {name}", fontsize=14,
            horizontalalignment='left',
            verticalalignment='center')

    ax.add_patch(
        Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                  height=18, facecolor=mc.CSS4_COLORS[name], edgecolor='0.7')
    )
  plt.draw()

  fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
  fig.subplots_adjust(margin/width, margin/height,
                      (width-margin)/width, (height-topmargin)/height)
  ax.set_xlim(0, cell_width * 4)
  ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)
  ax.set_axis_off()
  ax.set_title("Colours by luminance", fontsize=24, loc="left", pad=10)

  for i, zipp in enumerate(zip(duplicated_colours, duplicated_categories)):
    name, cat = zipp
    row = i % nrows
    col = i // nrows
    y = row * cell_height

    swatch_start_x = cell_width * col
    text_pos_x = cell_width * col + swatch_width + 7

    ax.text(text_pos_x, y, f"{cat} - {name}", fontsize=14,
            horizontalalignment='left',
            verticalalignment='center')

    ax.add_patch(
        Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                  height=18, facecolor=mc.CSS4_COLORS[name], edgecolor='0.7')
    )
  plt.draw()

  plt.show()

sort_colours()
Importing the custom drawio library:
- File --> Open Library From --> Device...
- Browse to library.xml file
- Library will appear in the left tool bar in drawio

Creating a custom library:
- File --> New Library --> Device...
- You can drag in custom images to the screen that pops up and then save the library
- You can also create custom objects in drawio and then group them together and drag them in to your library and resave
- You can also drag in .png images to drawio and then add them to your library that way

Notes:
- If you put x2 or x3 etc on a load, it will multiply the current by that number
- The source current should be left as a ? and will be calculated automatically
- The max current rating of a switching regulator, LDO, etc can be left as ?A w/out disturbing the current calculations. The sw freq can be left as ?KHz.
- Load currents must be defined, but 0A is allowed.
- Rails (lines) that aren't properly connected turn **red** automatically. A rail is flagged when it touches a component on one end but its other end is dangling (not snapped to a block) or loops back to the same component, so current can't back-propagate through it. Fix the connection and the line returns to its original color on the next recalculation. (Decorative/symbol lines and lines that don't touch any component are left alone.)

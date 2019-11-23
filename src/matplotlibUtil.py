
"""

@package matplotlibUtil A module for creating beautiful graphs

@copyright GNU Public License
@author written 2013, 2014 by Christian Herbst (www.christian-herbst.org) 
@author Sponsored by the Dept. of Cognitive Biology, University of Vienna, 
	
@note
This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 3 of the License, or (at your option) any later 
version.
@par
This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
@par
You should have received a copy of the GNU General Public License along with 
this program; if not, see <http://www.gnu.org/licenses/>.

"""

"""
usage example:

widthPx = 800
heightPx = 600
dpi = 100

graphLayout = matplotlibUtil.CGraph(
	width = widthPx / float(dpi),
	height = heightPx / float(dpi),
	dpi = dpi,
	lineWidth = 1.5,
	padding = 0.06,
	fontSize = 10,
	fontFamily = 'sans-serif',
	fontFace = 'Arial',
	fontWeight = 'normal'
)
arrRowRatios = [1, 1.5, 1, 1.5]
numPanels = len(arrRowRatios)
graphLayout.setRowRatios(arrRowRatios)
arrColumns = [1] * numPanels
#arrColumns[-1] = 2
graphLayout.setColumns(arrColumns)
fig = graphLayout.createFigure(adjustFont = True)
arrAx = graphLayout.arrAx

# do some plotting ...
ax = arrAx[0]
...


# finalize
for i, ax in enumerate(arrAx):
	ax.grid()
	ax.set_xlabel("Time [s]", labelpad=-2)
graphLayout.adjustPadding(
		left = 2.15, # factor by which self.padding is multiplied
		right = 1.2, # factor by which self.padding is multiplied
		top = 0.83, # factor by which self.padding is multiplied
		bottom = 1.1, # factor by which self.padding is multiplied
		hspace = 0.29, # hspace parameter for plt.subplots_adjust function
		wspace = 0.34, # wspace parameter for plt.subplots_adjust function
	)
graphLayout.addPanelNumbers(
		numeratorType = None,
		fontSize = 14,
		fontWeight = 'bold',
		countEveryPanel = True,
		format = '(%s)',
		offsetLeft = 0.09,
		offsetTop = 0.00,
		horizontalAlignment = 'center',
		verticalAlignment = 'top',
		arrAdditionalText = None,
		arrCustomNumerators = None,
		arrIgnore = None
	)
graphLayout.save("testGraph")

"""


import numpy
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.cm as cm
import generalUtility
import dspUtil
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patches as patches
from scipy import stats
import pylab

NUMERATOR_TYPE_ALPHA_LOWER = 0
NUMERATOR_TYPE_ALPHA_UPPER = 1
NUMERATOR_TYPE_ARABIC = 2	
NUMERATOR_TYPE_ROMAN = 3
NUMERATOR_TYPE_CUSTOM = 4

################################################################################

class CGraph:
	"""
	the main class, acting as a container for the graph
	"""
	def __init__(self,
		width = 8,
		height = 8,
		dpi = 72,
		lineWidth = 2,
		padding = 0.06,
		fontSize = 16,
		fontFamily = 'serif',
		fontFace = 'Times New Roman',
		fontWeight = 'normal',
		numeratorType = NUMERATOR_TYPE_ALPHA_UPPER
	
	):
		"""
		be default, the setup of the graph has one panel (i.e., only one row 
		and one column)
		@param width graph width [inches]
		@param height graph height [inches]
		@param dpi resolution, dots per inch (DPI)
		@param lineWidth graph line width
		@param padding graph padding [0..1]
		@param fontSize 
		@param fontFamily serif, sans-serif, etc.
		@param fontFace
		@param fontWeight normal, bold, etc...
		@param numeratorType numerator type for panels:
			NUMERATOR_TYPE_ALPHA_LOWER: lowercase letter
			NUMERATOR_TYPE_ALPHA_UPPER: uppercase letter
			NUMERATOR_TYPE_ARABIC: arabic numeral
			NUMERATOR_TYPE_ROMAN: roman numeral
		"""
		self.width = width
		self.height = height
		self.dpi = dpi
		self.lineWidth = lineWidth
		self.padding = padding
		self.fontSize = fontSize
		self.fontFamily = fontFamily
		self.fontSerif = fontFace
		self.fontWeight = fontWeight
		self.fig = None
		self.numeratorType = numeratorType
		self.arrAx = [] # the array that will store the individual panels
		self.arrAxPerRow = [] 
		self.arrRowRatios = []
		self.arrRowColumns = []
		
		self.setRowRatios([1])
	
	# ---------------------------------------------------------------------- #
	
	def setRowRatios(self,
		arrRowRatios
	):
		"""
		use this function to define how many panel rows contained in the graph,
		and what their respective relative panel height will be. calling this 
		function will reset any previously defined panel column information 
		(i.e., call setColumns immediately after the call to setRowRatios if 
		you want to have multi-column rows)
		
		@param arrRowRatios a list of relative row heights
		
		"""
		if not type(arrRowRatios).__name__ in ['list', 'ndarray']:
			raise Exception("arrRowRatios must either be a list or a numpy array")
		self.arrRowRatios = numpy.array(arrRowRatios)
		self.arrRowColumns = numpy.ones(len(arrRowRatios))
	
	# ---------------------------------------------------------------------- #
	
	def setColumns(self,
		arrColumnInfo
	):
		"""
		use this function if you need rows that contain more than one panel
		(i.e., multi-column rows). For practical reasons the column number needs
		to be restricted to 1 or a particular number of rows, such as, e.g.:
		[4, 1, 1, 4, 1]. Multiple numbers of colums (other than one) for
		different rows (such as, e.g., [4, 1, 2, 4, 1]) are not allowed
		
		@param a list containing information how many columns each row will have
		
		"""
		if not type(arrColumnInfo).__name__ in ['list', 'ndarray']:
			raise Exception("arrRowRatios must either be a list or a numpy array")
		if len(arrColumnInfo) != len(self.arrRowRatios):
			raise Exception("arrColumnInfo must have the same number of " \
				+ "entries as the previously defined self.arrRowRatios")
		for i in range(len(arrColumnInfo)):
			if arrColumnInfo[i] < 1:
				raise Exception("row " + str(i + 1) \
					+ " must have at least one column")
		numCols = None
		for n in arrColumnInfo:
			if n != int(n):
				raise Exception("column numbers need to be integers")
			if n != 1:
				if numCols is None:
					numCols = n
				else:
					if n != numCols:
						raise Exception("column information is too complex")
		self.arrRowColumns = numpy.array(arrColumnInfo)
		
	# ---------------------------------------------------------------------- #
		
	def createFigure(self, 
			adjustFont = True, widthRatios = None
	):
		"""
		creates the figure and takes care of the layout
		
		@param adjustFont if False, we'll take the system default font settings
		@param widthRatios if not None, an array need to be specified
		"""
		if len(self.arrRowRatios) < 1:
			raise Exception("need to define row ratios and columns first")
	
		if adjustFont:
			font = {
				'family' : self.fontFamily,
				'serif'  : self.fontSerif,
		        'weight' : self.fontWeight,
		        'size'   : self.fontSize,
			}
			matplotlib.rc('font', **font)
		#plt.close()
		plt.clf()
		self.fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
		
		self.arrAx = []
		self.arrAxPerRow = []
		numRows = len(self.arrRowRatios)
		numCols = int(self.arrRowColumns.max())
		gs = None
		if widthRatios is None:
			gs = gridspec.GridSpec(numRows, numCols,
			    height_ratios = self.arrRowRatios)
		else:
			gs = gridspec.GridSpec(numRows, numCols,
			    height_ratios = self.arrRowRatios, width_ratios = widthRatios)
		for row in range(numRows):
			numColsTmp = self.arrRowColumns[row]
			arrAxTmp = []
			if numColsTmp == 1:
				x = row * numCols
				y = row * numCols + (numCols - 1)
				ax = plt.subplot(gridspec.SubplotSpec(gs, x, y))
				self.arrAx.append(ax)
				arrAxTmp.append(ax)
			else:
				for col in range(numCols):
					ax = plt.subplot(gs[col + row * numCols])
					self.arrAx.append(ax)
					arrAxTmp.append(ax)
			self.arrAxPerRow.append(arrAxTmp)
		return self.arrAx
		
		
	# ---------------------------------------------------------------------- #
	
	def adjustPadding(self,
		left = 1.0, # factor by which self.padding is multiplied
		right = 1.0, # factor by which self.padding is multiplied
		top = 1.0, # factor by which self.padding is multiplied
		bottom = 1.0, # factor by which self.padding is multiplied
		hspace = 0.5, # hspace parameter for plt.subplots_adjust function
		wspace = 0.5, # wspace parameter for plt.subplots_adjust function
	):
		""" 
		convenience function to adjust padding in the graph
		
		@param left factor by which self.padding is multiplied
		@param right factor by which self.padding is multiplied
		@param top factor by which self.padding is multiplied
		@param bottom factor by which self.padding is multiplied
		@param hspace hspace parameter for plt.subplots_adjust function
		@param wspace wspace parameter for plt.subplots_adjust function
		"""
		plt.subplots_adjust(hspace=hspace, wspace=wspace, 
			bottom=self.padding * bottom, top=1.0 - self.padding * top, 
			left=self.padding * left, right = 1.0 - self.padding * right)
		
	# ---------------------------------------------------------------------- #

	def addPanelNumbers(self,
		numeratorType = None,
		fontSize = 14,
		fontWeight = 'bold',
		countEveryPanel = True,
		format = '(%s)',
		offsetLeft = 0.06,
		offsetTop = 0.00,
		horizontalAlignment = 'center',
		verticalAlignment = 'top',
		arrAdditionalText = None,
		arrCustomNumerators = None,
		arrIgnore = None
	):
		"""
		add numbers for the panels (top right corner). make sure this function
		is only called AFTER adjustPadding(...)
		
		@param numeratorType several numbering types are available: 
			- NUMERATOR_TYPE_ALPHA_LOWER: lowercase letter
			- NUMERATOR_TYPE_ALPHA_UPPER: uppercase letter
			- NUMERATOR_TYPE_ARABIC: arabic numeral
			- NUMERATOR_TYPE_ROMAN: roman numeral
			- NUMERATOR_TYPE_CUSTOM: need to specify via arrCustomNumerators
			If None, we'll use numerator type that was specified during object
			instantiation.
		@param fontSize
		@param fontWeight 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 
			'ultralight'
		@param countEveryPanel if False, only one label is added per row
		@param format format as used in the printf statement. 
		@param offsetLeft [0..1] offset of panel numbers from top-left \
			corner of axis
		@param offsetTop [0..1] offset of panel numbers from top-left \
			corner of axis
		@param horizontalAlignment 'center' | 'right' | 'left'
		@param verticalAlignment 'center' | 'top' | 'bottom' | 'baseline'
		@param arrAdditionalText if not None, we expect a list of n text
			items, where n is the number of panels that are numbered. The
			text is appended to each panel number label.
		@param arrCustomNumerators: custom numerators in a list. only used if
			numeratorType is NUMERATOR_TYPE_CUSTOM
		@param arrIgnore: either None (no effect), or a list with the zero-based
			indices of those panels for which no panel numbering should be 
			applied
		"""
		
		if numeratorType is None:
			numeratorType = self.numeratorType
		cnt = 0
		for rowIdx, arrAxTmp in enumerate(self.arrAxPerRow):
			for colIdx, ax in enumerate(self.arrAxPerRow[rowIdx]):
				if colIdx == 0 or countEveryPanel:

					additionalText = None
					if arrAdditionalText:
						additionalText = arrAdditionalText[cnt]
					customNumerator = None
					if arrCustomNumerators:
						customNumerator = arrCustomNumerators[cnt]

					doIgnore = False
					if not arrIgnore is None:
						if cnt in arrIgnore:
							doIgnore = True
					if not doIgnore:
						addPanelNumber(ax, cnt,
						    numeratorType=numeratorType,
							fontSize = fontSize,
							fontWeight = fontWeight,
							format = format,
							offsetLeft = offsetLeft,
							offsetTop = offsetTop,
							horizontalAlignment = horizontalAlignment,
							verticalAlignment = verticalAlignment,
							additionalText = additionalText,
							customNumerator = customNumerator
						)
					cnt += 1

					# s = ''
					# if numeratorType == NUMERATOR_TYPE_ALPHA_LOWER:
					# 	s = chr(96 + cnt)
					# elif numeratorType == NUMERATOR_TYPE_ALPHA_UPPER:
					# 	s = chr(64 + cnt)
					# elif numeratorType == NUMERATOR_TYPE_ARABIC:
					# 	s = str(cnt)
					# elif numeratorType == NUMERATOR_TYPE_ROMAN:
					# 	s = generalUtility.intToRoman(cnt)
					# elif numeratorType == NUMERATOR_TYPE_CUSTOM:
					# 	s = arrCustomNumerators[cnt - 1]
					# else:
					# 	raise Exception("inavlid numerator type")
					# label = format % s
					# if not arrAdditionalText is None:
					# 	label += arrAdditionalText[cnt-1]
					# bbox = ax.get_position()
					# x = bbox.x0 - offsetLeft
					# y = bbox.y0 + bbox.height + offsetTop
					# doIgnore = False
					# if not arrIgnore is None:
					# 	if cnt - 1 in arrIgnore:
					# 		doIgnore = True
					# if not doIgnore:
					# 	plt.figtext(x, y, label, size=fontSize, weight=fontWeight,
					# 		ha = horizontalAlignment, va = verticalAlignment)

	# ---------------------------------------------------------------------- #
	
	def getArrAx(self):
		return self.arrAx
	
	# ---------------------------------------------------------------------- #
		
	def getFig(self):
		return self.fig	
	
	# ---------------------------------------------------------------------- #

	def save(self, fNameBase):
		saveGraph(fNameBase)

	# ---------------------------------------------------------------------- #
	

################################################################################

def createGraph(width, height, arrGraphRowRatios, arrGraphColumns, 
		fontSize = 11, fontSerif = True
	):
	"""
	convenience function to create a graph
	@return the instantiated CGraph object, and a list with all axes
	"""
	plt.clf()
	fontFamily = 'sans-serif'
	fontFace = 'Arial'
	if fontSerif:
		fontFamily = 'serif'
		fontFace = 'Times New Roman'
	graph = CGraph(
		width = width,
		height = height,
		dpi = 100,
		lineWidth = 1.3,
		padding = 0.025,
		fontSize = fontSize,
		fontFamily = fontFamily,
		fontFace = fontFace,
		fontWeight = 'normal'
	)
	graph.setRowRatios(arrGraphRowRatios)
	graph.setColumns(arrGraphColumns)
	arrAx = graph.createFigure()
	return graph, arrAx

################################################################################

def saveGraph(fNameOnly):
	"""
	utility function to save the graph as both png and svg file
	@param fNameOnly the output file name (including the path) MINUS THE SUFFIX,
		e.g. /Users/ch/Desktop/shinyNewGraph
	"""
	for dType in ['png', 'svg']:
		plt.savefig(fNameOnly + '.' + dType)

################################################################################

def addPanelNumber(
		ax,
		index, # zero-based
		numeratorType = None,
		fontSize = 14,
		fontWeight = 'bold',
		format = '(%s)',
		offsetLeft = 0.06,
		offsetTop = 0.00,
		horizontalAlignment = 'center',
		verticalAlignment = 'top',
		additionalText = None,
		customNumerator = None
	):

	s = ''
	if numeratorType == None:
		numeratorType = NUMERATOR_TYPE_ALPHA_UPPER
	if numeratorType == NUMERATOR_TYPE_ALPHA_LOWER:
		s = chr(96 + index + 1)
	elif numeratorType == NUMERATOR_TYPE_ALPHA_UPPER:
		s = chr(64 + index + 1)
	elif numeratorType == NUMERATOR_TYPE_ARABIC:
		s = str(index + 1)
	elif numeratorType == NUMERATOR_TYPE_ROMAN:
		s = generalUtility.intToRoman(index + 1)
	elif numeratorType == NUMERATOR_TYPE_CUSTOM:
		s = customNumerator
	else:
		raise Exception("inavlid numerator type")
	label = format % s
	if not additionalText is None:
		label += additionalText
	bbox = ax.get_position()
	x = bbox.x0 - offsetLeft
	y = bbox.y0 + bbox.height + offsetTop
	plt.figtext(x, y, label, size=fontSize, weight=fontWeight,
		ha = horizontalAlignment, va = verticalAlignment)

	# ---------------------------------------------------------------------- #

def formatAxisTicks(
		ax,
		axisType,
		majorAxisDivisor,
		format='%-1.2f',
		minorAxisRelativeDivisor = 5.0
	): 
	"""
	change the format to the axis ticks
	@param ax matplotlib axes object
	@param axisType 0 or 'x': xaxis; 1 or 'y': yaxis
	@param majorAxisDivisor the major axis divisor
	@param format formatting string
	@param minorAxisRelativeDivisor value by which the major axis divisor will 
		be divided in order to obtain the minor axis divisor
	"""
	majorLocator   = MultipleLocator(majorAxisDivisor)
	majorFormatter = FormatStrFormatter(format)
	minorLocator   = MultipleLocator(majorAxisDivisor \
		/ float(minorAxisRelativeDivisor))
	if axisType == 0 or axisType == 'x':
		ax.xaxis.set_major_locator(majorLocator)
		ax.xaxis.set_major_formatter(majorFormatter)
		ax.xaxis.set_minor_locator(minorLocator)
	elif axisType == 1 or axisType == 'y':
		ax.yaxis.set_major_locator(majorLocator)
		ax.yaxis.set_major_formatter(majorFormatter)
		ax.yaxis.set_minor_locator(minorLocator)
	else:
		raise Exception("axis type not recignized (must be either 0 [xaxis] " \
			+ "or 1 [yaxis]")
		
################################################################################

def setLimit(
		ax, 
		arrData = None,
		axis = 'x',
		rangeMultiplier = 0.1
	):
	"""
	sets the x-axis limit of a graph centred on the respective data values, 
	adding some extra headspace on both the minimum and maximum end
	@param ax matplotlib axes object
	@param arrData either a list or a 1D numpy array (from which we get the \
		min/max values). If None, we'll take the current value limits of the
		given ax
	@param axis either x-axis ('x' or 0) or y-axis ('y' or 1)
	@param rangeMultiplier leeway (extending the data range) on both ends of 
		the y axis
	"""
	if arrData is None:
		if axis == 'x':
			arrData = ax.get_xlim()
		elif axis == 'y':
			arrData = ax.get_ylim()
	valMin = min(arrData)
	valMax = max(arrData)
	valRange = valMax - valMin
	leeway = valRange * rangeMultiplier
	if axis in ['x', 0]:
		ax.set_xlim(valMin - leeway, valMax + leeway)
	elif axis in ['y', 1]:
		ax.set_ylim(valMin - leeway, valMax + leeway)
	else:
		raise Exception("undefined axis")

################################################################################

def plotOnTwoScales(ax, arrX, arrY1, arrY2, label1, label2, col1, col2,
        linewidth = 1, arrX2 = None):
	"""
	plot two time-series with two different y-axis scales (with two different
	colors), one scale on the left and the other on the right side of the graph
	@param ax: the graph canvas
	@param arrX: the shared x-axis data (typically a time-series)
	@param arrY1: the y-axis data for the first graph
	@param arrY2: the y-axis data for the second graph
	@param label1: the label for the first graph
	@param label2: the label for the second graph
	@param col1: the color for the first graph
	@param col2: the color for the second graph
	@param linewidth: the linewidth for plotting
	@param arrX2: if a second x-axis data array is specified, this is being
		used for plotting the 2nd data series (if None, we'll use the common
		x-axis data array, specified by parameter arrX)
	@return: the twin-ax
	"""
	ax.plot(arrX, arrY1, color=col1, linewidth=linewidth)
	ax.set_ylabel(label1, color=col1)
	ax.tick_params('y', colors=col1)
	ax2 = ax.twinx()
	if arrX2 is None:
		arrX2 = arrX
	ax2.plot(arrX2, arrY2, color=col2, linewidth=linewidth)
	ax2.set_ylabel(label2, color=col2)
	ax2.tick_params('y', colors=col2)
	return ax2

################################################################################

def plotPolyFit(ax, dataX, dataY, degrees = 1, fontSize = 10, lineSize = 3, 
		lineColor = 'red', txtX = None, txtY = None, numDigitsEq = 4, lineStyle = 'k--', verbose = False
):
	"""
	performs a polynomial fit through the given data
	@param ax matplotlib axes object
	@param dataX x-axis data array. must not contain NaN or Inf values
	@param dataY y-axis data array. must not contain NaN or Inf values
	@param degrees degrees of the fitted polynomial
	@param fontSize font size for printing the polynomial equation
	@param lineSize line width of the fit
	@param lineColor color of the fitted polynomial on the graph
	@param txtX x-axis co-ordinate (data value) of the equation 
	@param txtY y-axis co-ordinate (data value) of the equation  
	@param numDigitsEq numer of decimal digits in the printed equation
	"""
	
	if len(dataX) < 2:
		raise Exception("data count must be 2 or larger")
		
	#if dspUtil.containsNanInf(dataX):
	#	raise Exception("x-axis data contains NaN or Inf values. aborting.")
	#if dspUtil.containsNanInf(dataY):
	#	raise Exception("y-axis data contains NaN or Inf values. aborting.")
	
	
	# remove NaN and Inf values
	dataX, dataY = generalUtility.removeNanInf(dataX, dataY)
		
	p = numpy.polyfit(dataX, dataY, degrees)
	#print "\t\t", p
	y1 = numpy.polyval(p,dataX)
	#plt.plot(x, y1, "k-", color='red', linewidth = 2)
	def getY(p, x):
		y = 0
		for i in range(len(p)):
			y += p[len(p) - (i + 1)] * (x ** float(i))
		return y
	arrX = []
	arrY = []
	xMin = min(dataX)
	xMax = max(dataX)
	step = (xMax - xMin) / 1000.0
	for x in numpy.arange(xMin, xMax, step):
		y = getY(p, x)
		arrX.append(x)
		arrY.append(y)
	ax.plot(arrX, arrY, lineStyle, color=lineColor, linewidth = lineSize)	

	# calculate the variance and R2
	SSerr = 0
	SStot = 0
	mean = sum(dataY) / float(len(dataY))
	for i in range(len(dataX)):
		y_i = dataY[i]
		f_i = getY(p, dataX[i])
		SSerr += (y_i - f_i) * (y_i - f_i)
		SStot += (y_i - mean) * (y_i - mean)
	R2 = 1 - (SSerr / SStot)
	#print "\t\tR2:", R2

	if fontSize > 0:
		txt = '$y = '
		for i in range(len(p)):
			factor = 1
			if i > 0:
				if p[i] > 0:
					txt += '+ '
				else:
					txt += '- '
					factor = -1
			txt += (('%.' + str(numDigitsEq) + 'f ') % (p[i] * factor))
			idx = len(p) - (i + 1)
			if idx == 0:
				pass
			elif idx == 1:
				txt += 'x '
			else: txt += ' x^%d '%idx
		txt += (('$\n$R^2 = %.' + str(numDigitsEq) + 'f$') % R2)
		if verbose: print txt
		yMin = min(dataY)
		yMax = max(dataY)
		if txtX is None:
			txtX = xMin + (xMax - xMin) / 20.0
		if txtY is None:
			txtY = yMin + (yMax - yMin) / 25.0
			if len(p) == 2 and p[0] > 0:
				txtY = yMax - (yMax - yMin) / 6.0
		ax.text(txtX, txtY, txt, fontsize = fontSize)
	#print "\t\tdone"
	return p, R2, xMin, xMax

################################################################################

def plotIsocontours(
	ax, # the axis object on which we'll plot
	arrX, # x-axis grid values, 1D vector
	arrY, # y-axis grid values, 1D vector
	arrZ, # z-axis data values, 2D array
	colorMap = None, # colour map
	numIsocontours = 6, # number of isocountours
	paintAlpha = 0.75, 
	contourFontSize = 0, # if 0 or None, we won't add contour lables
	isoContourColor = 'black', 
	isoContourLineWidth = 0.5, 
	isoContourAlpha = 0.75
):
	""" 
	draw a 2D grid with respective values (i.e., a 3D data object) and add
	isocontour lines
	@param ax the axis object on which we'll plot
	@param arrX x-axis grid values, 1D vector
	@param arrY y-axis grid values, 1D vector
	@param arrZ z-axis data values, 2D array
	@param colorMap colour map
	@param numIsocontours number of isocountours
	@param paintAlpha 
	@param contourFontSize if 0 or None, we won't add contour lables
	@param isoContourColor 
	@param isoContourLineWidth 
	@param isoContourAlpha 
	"""
	CS = ax.contour(arrX, arrY, arrZ, numIsocontours, colors = isoContourColor, \
		linewidth = isoContourLineWidth, alpha = isoContourAlpha)
	#plt.pcolormesh(x, y, z, cmap = plt.get_cmap('rainbow'))
	if colorMap is None: colorMap = cm.afmhot
	CS2 = ax.contourf(arrX, arrY, arrZ, numIsocontours, \
		alpha = isoContourAlpha, cmap = colorMap)
	if contourFontSize > 0:
		ax.clabel(CS, inline = 1, fontsize = contourFontSize)

################################################################################

def plotSpectrogram(
		ax, 
		data, 
		fs, 
		windowSize = 4096, 
		readProgress = 0.05, 
		fMax = 5000, 
		dynamicRange = 50,
		spectrumType = dspUtil.POWER_SPECTRUM,
		cmap='gray'
	):
	""" 
	convenience function for generating a spectrogram (by calling 
	@ref calculateSpectrogram() from the @ref dspUtil module, and plotting the 
	contents at the same time
	@ax matplotlib axes object, on which the spectrogram is plotted
	@param data a list or numpy array
	@param fs sampling frequency [Hz]
	@param windowSize see @ref calculateFFT()
	@param readProgress time interval by which the analysis window is advanced.
	@param fMax maximum frequency [Hz]. no data above this frequency is being
		returned by this function (even though it may be calculated, depending
		on the sampling frequency), as a convenience for plotting the results
	@param dynamicRange inidcated in dB. similar as in Praat, see 
		<a href="http://www.fon.hum.uva.nl/praat/manual/Intro_3_2__Configuring_the_spectrogram.html">there</a>
	@param spectrumType see @ref calculateFFT()
	@cmap color map
	@return a three-dimensional array [y][x][z] ready to be plotted with the
		matplotlib <a href="
http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow">imshow</a>
		function. Note that the third (i.e., z) dimension is set to grayscale,
		i.e., all three RGB values have the same value, which is a bit redundant.
	"""
	spectrogramData = dspUtil.calculateSpectrogram(data, fs, windowSize, \
		readProgress, fMax, dynamicRange, spectrumType)
	duration = len(data) / float(fs)
	ax.imshow(spectrogramData, aspect='auto', origin='lower', 
		extent=[0, duration,0,fMax], cmap=cmap)
	return spectrogramData

################################################################################
	
def plotData(
	ax,
	data, 
	fs,
	labelY,
	tStart = 0, 
	color='blue',
	linewidth = 1, 
	valMin = None,
	valMax = None,
	alpha = 0.8,
	dontPlotData = False
):
	"""
	convenience function to plot time-series data. no rocket science here, just
	emulating the funcionality of a few function calls in one single function
	@param ax matplotlib axes object, onto which the graph is drawn
	@param data a numpy array or a list of numpy arrays
	@param fs sampling freuquency [Hz]
	@param labelY a string containing the label of the y axis
	@param tStartOffset useful when the time axis should not start at t = 0, but
		at a later moment in time
	@param color graph colour
	@param linewidth graph line width
	@param valMin minimum display value. if None, we'll take the minimum of the
		provided data
	@param valMax maximum display value. if None, we'll take the maximum of the
		provided data
	@param alpha transparency of the graph [0..1]
	@param dontPlotData for some very large files it might be useful to only 
		plot labels and grid, but not the actual data, if the graph is 
		imported into Adobe Illustrator (would take forever to load)
	@return nothing
	"""
	arrData = None
	dataType = type(data).__name__
	if dataType == 'list':
		arrData = data
	else:
		arrData = [data]
	numFrames = len(arrData[0])
	dataT = numpy.zeros(numFrames)
	for i in range(numFrames):
		dataT[i] = float(i) / float(fs)
	for data in arrData:
		ax.plot(dataT + tStart, data, linewidth=linewidth, alpha=alpha, color=color)
	ax.grid()
	if (not valMin is None) and (not valMax is None):
		ax.set_ylim(valMin, valMax)
	ax.set_xlim(tStart, tStart + dataT[-1])
	ax.set_xlabel('Time [s]')
	ax.set_ylabel(labelY)
	return
	
################################################################################
	
def plotVerticalMarkers(
	ax, arrMarkerOffset,
	color = 'red',
	valMin = -1, 
	valMax = 1,
	linestyle = '--',
	linewidth = 1,
	alpha = 1
):
	"""
	plot vertical markers on a graph, e.g. indicating the temporal offset of
	an interesting event, or highlighting data extraction
	@param ax matplotlib axes object
	@param arrMarkerOffset x-axis value
	@param color a matplotlib colour
	@param valMin minimum value of the graph
	@param valMax maximum value of the graph
	@param linestyle a valid matplotlib linestyle
	@param linewidth line width
	@param alpha transparency setting [0..1]
	"""
	dataType = type(arrMarkerOffset).__name__
	if dataType != 'list':
		arrMarkerOffset = [arrMarkerOffset]
	for markerOffset in arrMarkerOffset:
		ax.plot([markerOffset, markerOffset], [valMin, valMax], linestyle, \
			color=color, linewidth = linewidth, alpha = alpha)
	
################################################################################

def drawRectangle(ax, t1, t2, borderColor, fillColor, fillAlpha = 0.1,
        borderWidth = 1, yMin = None, yMax = None):
	"""
	draw a semi-transparent rectangle on a graph
	:param ax: matplotlib axes object
	:param t1: start time on x axis
	:param t2: end time of x axis
	:param borderColor: border color
	:param fillColor: fill color
	:param fillAlpha: opacity indicator for fill color
	:param borderWidth: width of border
	:param yMin: minimum y-axis value. if None, we'll get this from current extremes
	:param yMax: maximum y-axis value. if None, we'll get this from current extremes
	:return:
	"""
	if yMin is None:
		if not yMax is None:
			raise Exception("if yMin is None, yMax must also be None")
		else:
			yMin, yMax = ax.get_ylim()
	else:
		if yMax is None:
			raise Exception("if yMax is None, yMin must also be None")

	rect = patches.Rectangle((t1,yMin), t2-t1, yMax-yMin,
	    linewidth=borderWidth, edgecolor=borderColor, facecolor=fillColor, alpha=fillAlpha)
	ax.add_patch(rect)

################################################################################

def plotPixelPreciseImage(img, dpi, outputFileName, cmap = 'gray'):
	"""
	creates a PNG image at the exact pixel dimensions of the given image data.
	only the specified image data is plotted, all axes are removed. Note that 
	this functin creates a new figure. Therefore, DO NOT CALL THIS DURING 
	GENERATION OF ANOTHER GRAPH.
	@param img a 2D or 3D array (depending on colour depth), where y is the first
		and x is the 2nd dimension (as e.g. returned by matplotlib's imread function)
	@param dpi dots per inch (typically either 100 or 72)
	@param outputFileName full output file name (including path and suffix)
	@param cmap the cmap parameter of matplotlib's imshow function
	"""
	
	plt.close()
	heightPx = len(img)
	widthPx = len(img[0])
	heightInch = float(heightPx) / float(dpi)
	widthInch = float(widthPx) / float(dpi)
	fig = plt.figure(figsize=(widthInch, heightInch))
	ax = plt.subplot(111)
	ax.imshow(img, cmap=cmap, aspect = 'equal', interpolation='none')
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)
	plt.savefig(outputFileName)

################################################################################

def plotVectorData(vectorData, ax, offsetX, offsetY, height = None, width = None,
		flipVertically = True, facecolor=(0.6, 0.6, 0.6), edgecolor=(0.0, 0.0, 0.0)):
	"""
	utility function to plot SVG vector graphics data
	@param vectorData: SVG vectorData (when using XML data
		saved from a Gimp path, take the data from the 'd'
		attribute in the 'path' node
	@param ax: the axis on which to plot
	@param offsetX: the x offset within the axis
	@param offsetY: the y offset within the axis
	@param height: the height to which the SVG data structure should be
		scaled. If None, we won't scale the structure
	@param width: the width to which the SVG data structure should be
		scaled. If None, we'll choose this proportional to the height parameter.
	@param flipVertically: if True, we'll flip the sctructure upside down
	@return: a list containing the dimensions of the structure [xMin, xMax, yMin, yMax]
	"""
	arrVertices = []
	arrCodes = []
	parts = vectorData.split()
	i = 0
	code_map = {
		'M': (Path.MOVETO, 1),
		'C': (Path.CURVE4, 3),
		'L': (Path.LINETO, 1),
		'Z': (None, 0)
	}

	arrXall = []
	arrYall = []
	previousCode = None

	# parse the data structure
	while i < len(parts):
		code = parts[i]
		counterModifier = 1
		if not code in code_map:
			code = previousCode
			counterModifier = 0
		previousCode = code
		path_code, npoints = code_map[code]
		arrCodes.extend([path_code] * npoints)
		#print i, code, path_code, npoints, counterModifier, path_code * npoints
		arrDataPoints = parts[i + counterModifier:i + npoints + counterModifier]
		for data in arrDataPoints:
			tmp = data.split(',')
			arrXall.append(float(tmp[0]))
			arrYall.append(float(tmp[1]))
		#print i, code, npoints, counterModifier, arrDataPoints
		#print "\t", [[float(x) for x in y.split(',')] for y in arrDataPoints]
		arrVertices.extend([[float(x) for x in y.split(',')] for y in arrDataPoints])
		i += npoints + counterModifier
	arrVertices = numpy.array(arrVertices, numpy.float)
	arrYall = numpy.array(arrYall, dtype=numpy.float32)
	arrXall = numpy.array(arrXall, dtype=numpy.float32)
	yMin = arrYall.min()
	yMax = arrYall.max()
	xMin = arrXall.min()
	xMax = arrXall.max()
	yRange = yMax - yMin
	xRange = xMax - xMin

	# flip structure upside down?
	if flipVertically:
		for i, data in enumerate(arrVertices):
			#print i, data
			x = data[0]
			y = data[1]
			y = yMin + yRange - (y - yMin)
			arrVertices[i] = [x, y]

	# scale the structure?
	if height:
		if width:
			pass
		else:
			width = float(xRange) * float(height) / float(yRange)
		scaleX = float(width) / float(xRange)
		scaleY = float(height) / float(yRange)
		for i, data in enumerate(arrVertices):
			#print i, data
			x = data[0] * scaleX
			y = data[1] * scaleY
			arrVertices[i] = [x, y]

	# add offsets and get final extensions
	n = len(arrVertices)
	arrXall = numpy.zeros(n)
	arrYall = numpy.zeros(n)
	for i in range(n):
		x = arrVertices[i][0] + offsetX
		y = arrVertices[i][1] + offsetY
		arrXall[i] = x
		arrYall[i] = y
		arrVertices[i] = [x, y]

	# finally, plot the data structure
	vectorData_path = Path(arrVertices, arrCodes)
	vectorData_patch = PathPatch(vectorData_path, facecolor= facecolor,
		edgecolor= edgecolor)
	ax.add_patch(vectorData_patch)

	return [arrXall.min(), arrXall.max(), arrYall.min(), arrYall.max()]

################################################################################

def plotMusicalStaff(ax, arrT, arrF0, middleC = 261.6255653, linewidth=2,
        markersize = 3, fontSize = 11, color='darkblue', omitClefs = False):
	"""
	utility function to plot fundamental frequency traces on a piano staff
	@param ax:
	@param arrT:
	@param arrF0:
	@param middleC: frequency of middle C (C4) [Hz]
	@return: tMin and tMax of entire graph
	"""

	# check data
	n = len(arrT)
	if len(arrF0) != n:
		raise Exception("array length don't match")
	tStartSignal = arrT[0]
	tEnd = arrT[-1]
	durationSignal = tEnd - tStartSignal
	tStartStaff = tStartSignal - durationSignal * 0.15
	durationTotal = tEnd - tStartStaff
	if durationSignal <= 0:
		raise Exception("duration must not be zero or negative")

	# convert to cents
	arrOct = numpy.zeros(n)
	for i in range(n):
		oct = numpy.nan
		f0 = arrF0[i]
		if f0 > 0:
			oct = dspUtil.hertzToCents(arrF0[i], middleC) / 1200.0
		arrOct[i] = oct

	# plot staff lines
	staffLineDistanceFromC4 = 2.0 / 14.0 # in octaves
	for i in range(-10, 12, 2):
		if i != 0:
			c = i * staffLineDistanceFromC4
			plt.plot([tStartStaff, tEnd], [c, c], color='black', linewidth=0.5)

	# plot the clefs
	if not omitClefs:
		height = 1.8
		xMin, xMax, yMin, yMax = plotVectorData(violinClefVectorData, ax,
			offsetY = -0.1, offsetX = tStartSignal - durationSignal * 0.12,
			height = height, width = durationSignal * 0.05, facecolor=(0, 0, 0))
		height = 0.8
		xMin, xMax, yMin, yMax = plotVectorData(bassClefVectorData, ax,
			offsetY = -1.14, offsetX = tStartSignal - durationSignal * 0.12,
			height = height, width = durationSignal * 0.06, facecolor=(0, 0, 0))

	# plot data
	if linewidth > 0:
		ax.plot(arrT, arrOct, linewidth = linewidth, color=color)
	if markersize > 0:
		ax.plot(arrT, arrOct, '.', markersize = markersize, color=color)
	ax.set_xlim(tStartStaff, tEnd)
	valMin = numpy.nanmin(arrOct)
	valMax = numpy.nanmax(arrOct)
	if valMin > -2: valMin = -2
	if valMax < 2: valMax = 2
	ax.set_ylim(valMin, valMax)
	ax.set_ylabel("Octaves vs. C4\n($f \\approx$ %.1f Hz)" % middleC,
	            size=fontSize)
	ax.set_xlabel("Time [s]", size=fontSize)

	#a.plot(t, np.sin((i + 1) * 2 * np.pi * t))
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	return [tStartStaff, tEnd]

################################################################################

violinClefVectorData = """
           M 132.51,973.64
           C 141.16,980.03 153.68,983.57 164.30,984.63
             164.30,984.63 173.74,985.52 173.74,985.52
             173.74,985.52 178.46,985.52 178.46,985.52
             178.46,985.52 181.93,985.73 181.93,985.73
             181.93,985.73 186.02,985.52 186.02,985.52
             192.59,985.49 201.54,983.23 207.74,981.02
             226.10,974.46 243.13,961.53 254.67,945.86
             257.53,941.98 260.82,936.78 262.63,932.33
             268.18,918.62 268.19,902.51 268.17,887.94
             268.17,887.94 267.85,882.91 267.85,882.91
             267.85,882.91 267.85,881.65 267.85,881.65
             267.85,881.65 267.54,878.50 267.54,878.50
             267.54,878.50 267.54,875.67 267.54,875.67
             267.54,875.67 266.65,866.54 266.65,866.54
             266.65,866.54 265.56,856.15 265.56,856.15
             265.56,856.15 263.58,843.88 263.58,843.88
             263.58,843.88 258.14,809.57 258.14,809.57
             258.14,809.57 252.87,776.20 252.87,776.20
             252.87,776.20 251.56,768.02 251.56,768.02
             251.44,767.23 251.25,765.28 250.67,764.77
             250.11,764.28 249.02,764.51 248.34,764.60
             248.34,764.60 242.04,765.49 242.04,765.49
             242.04,765.49 228.82,766.76 228.82,766.76
             228.82,766.76 221.58,767.08 221.58,767.08
             221.58,767.08 216.86,767.39 216.86,767.39
             216.86,767.39 197.03,767.39 197.03,767.39
             197.03,767.39 192.31,767.08 192.31,767.08
             192.31,767.08 184.44,766.76 184.44,766.76
             184.44,766.76 181.30,766.47 181.30,766.47
             181.30,766.47 169.02,765.28 169.02,765.28
             158.84,763.95 148.76,762.08 139.12,758.44
             129.51,754.82 112.62,745.45 103.55,740.01
             75.83,723.37 50.83,702.70 32.53,675.79
             18.07,654.55 9.75,629.48 5.22,604.34
             3.77,596.32 2.40,588.23 1.63,580.11
             1.63,580.11 1.21,574.44 1.21,574.44
             1.21,574.44 0.32,564.68 0.32,564.68
             0.32,564.68 0.32,559.65 0.32,559.65
             0.32,559.65 0.00,554.93 0.00,554.93
             0.00,554.93 0.32,545.17 0.32,545.17
             0.32,545.17 0.64,542.02 0.64,542.02
             0.64,542.02 0.64,539.19 0.64,539.19
             0.64,539.19 3.33,520.62 3.33,520.62
             7.64,496.72 16.78,473.11 28.08,451.68
             33.36,441.68 38.90,431.80 45.24,422.41
             55.31,407.53 66.60,393.53 79.01,380.55
             79.01,380.55 84.35,375.50 84.35,375.50
             84.35,375.50 89.70,370.19 89.70,370.19
             89.70,370.19 95.37,365.10 95.37,365.10
             106.32,354.91 117.59,345.02 129.05,335.41
             129.05,335.41 160.21,310.32 160.21,310.32
             160.21,310.32 169.65,303.01 169.65,303.01
             170.54,302.31 173.65,300.07 174.09,299.28
             174.71,298.21 173.73,294.41 173.51,293.04
             173.51,293.04 170.99,276.36 170.99,276.36
             170.99,276.36 168.62,258.11 168.62,258.11
             168.62,258.11 167.19,246.77 167.19,246.77
             167.19,246.77 166.16,231.35 166.16,231.35
             166.16,231.35 165.24,216.56 165.24,216.56
             165.24,216.56 164.94,213.41 164.94,213.41
             164.94,213.41 164.94,209.95 164.94,209.95
             164.94,209.95 164.61,205.23 164.61,205.23
             164.61,205.23 164.61,199.56 164.61,199.56
             164.61,199.56 164.33,195.15 164.33,195.15
             164.33,195.15 164.61,191.38 164.61,191.38
             164.61,191.38 164.61,181.93 164.61,181.93
             164.61,181.93 164.93,176.58 164.93,176.58
             164.93,176.58 165.53,166.19 165.53,166.19
             165.53,166.19 165.93,158.33 165.93,158.33
             165.93,158.33 167.40,145.11 167.40,145.11
             168.35,136.15 170.51,121.53 172.85,113.00
             179.31,89.37 191.87,66.04 204.77,45.33
             212.35,33.16 222.38,19.04 233.23,9.72
             239.44,4.38 246.52,-0.10 254.95,0.00
             259.27,0.05 263.13,2.70 266.28,5.44
             272.43,10.79 279.64,22.79 283.35,30.22
             294.08,51.66 302.49,74.18 308.90,97.26
             311.89,108.00 314.25,119.22 315.43,130.31
             315.43,130.31 316.32,139.75 316.32,139.75
             316.32,139.75 316.64,145.73 316.64,145.73
             316.64,145.73 316.64,160.84 316.64,160.84
             316.64,160.84 316.33,165.56 316.33,165.56
             316.30,179.45 314.07,198.49 311.29,212.15
             306.90,233.66 300.36,249.59 290.59,269.12
             283.16,283.99 276.87,295.32 266.95,308.78
             266.95,308.78 248.28,333.02 248.28,333.02
             248.28,333.02 238.24,345.29 238.24,345.29
             234.09,349.75 229.61,353.82 225.05,357.84
             225.05,357.84 217.49,363.93 217.49,363.93
             217.49,363.93 208.68,370.97 208.68,370.97
             208.68,370.97 204.27,374.44 204.27,374.44
             203.62,374.96 202.57,375.68 202.35,376.51
             202.20,377.07 202.47,378.37 202.57,378.97
             202.57,378.97 203.57,384.64 203.57,384.64
             203.57,384.64 206.98,404.78 206.98,404.78
             206.98,404.78 215.68,456.09 215.68,456.09
             215.68,456.09 218.46,472.46 218.46,472.46
             218.46,472.46 219.70,480.33 219.70,480.33
             219.70,480.33 230.40,479.38 230.40,479.38
             230.40,479.38 242.67,479.38 242.67,479.38
             242.67,479.38 246.45,479.70 246.45,479.70
             246.45,479.70 249.28,479.70 249.28,479.70
             249.28,479.70 254.95,480.27 254.95,480.27
             275.23,482.21 294.90,487.22 311.60,499.41
             341.31,521.10 358.87,552.28 364.66,588.29
             365.61,594.21 366.68,603.16 366.69,609.07
             366.69,609.07 367.00,614.42 367.00,614.42
             367.00,614.42 367.00,621.03 367.00,621.03
             367.00,621.03 366.69,626.06 366.69,626.06
             366.67,635.86 363.09,652.87 359.96,662.26
             352.40,684.96 338.10,707.50 320.99,724.18
             315.89,729.15 310.36,733.64 304.68,737.92
             295.59,744.77 284.24,751.70 273.83,756.33
             273.83,756.33 264.39,760.47 264.39,760.47
             264.39,760.47 268.25,786.91 268.25,786.91
             268.25,786.91 275.18,833.18 275.18,833.18
             275.18,833.18 278.78,856.78 278.78,856.78
             278.78,856.78 281.38,879.76 281.38,879.76
             281.38,879.76 281.38,882.59 281.38,882.59
             281.38,882.59 282.02,891.41 282.02,891.41
             282.02,891.41 282.02,906.83 282.02,906.83
             282.02,906.83 281.73,909.98 281.73,909.98
             281.19,917.91 280.07,925.74 277.39,933.27
             272.91,945.86 262.11,960.42 252.97,970.07
             252.97,970.07 250.54,972.31 250.54,972.31
             246.79,975.90 242.85,979.12 238.58,982.07
             225.36,991.18 210.38,996.57 194.52,998.78
             194.52,998.78 181.93,1000.00 181.93,1000.00
             181.93,1000.00 171.85,1000.00 171.85,1000.00
             171.85,1000.00 168.08,999.67 168.08,999.67
             168.08,999.67 165.24,999.67 165.24,999.67
             159.48,999.08 153.59,998.28 147.93,997.03
             137.70,994.78 127.85,991.24 118.35,986.88
             101.43,979.11 86.42,969.37 76.93,952.79
             72.20,944.51 69.60,935.19 68.90,925.72
             68.90,925.72 68.58,922.25 68.58,922.25
             68.58,922.25 68.58,916.27 68.58,916.27
             68.58,916.27 68.58,913.44 68.58,913.44
             68.58,913.44 68.58,910.61 68.58,910.61
             68.65,907.65 69.44,902.57 70.00,899.59
             73.57,880.50 85.54,861.15 103.87,853.28
             114.41,848.75 128.54,847.25 139.75,849.75
             154.98,853.16 171.10,865.52 178.06,879.45
             181.48,886.27 182.70,892.39 183.21,899.91
             183.21,899.91 183.50,904.94 183.50,904.94
             183.45,908.79 182.67,914.78 181.61,918.48
             180.36,922.85 178.77,927.14 176.43,931.07
             165.26,949.86 143.76,958.89 123.70,964.75
             126.71,968.22 128.69,970.81 132.51,973.64 Z
           M 192.94,284.10
           C 192.94,284.10 200.81,277.62 200.81,277.62
             209.05,270.51 217.03,263.29 224.73,255.59
             242.77,237.55 259.51,217.79 273.02,196.10
             281.95,181.76 290.82,164.12 293.53,147.31
             294.53,141.05 294.68,136.60 294.61,130.31
             294.61,130.31 293.71,121.18 293.71,121.18
             292.13,107.91 287.67,91.49 274.46,85.40
             269.65,83.18 265.46,83.04 260.30,83.10
             256.41,83.14 251.04,84.42 247.39,85.80
             228.94,92.77 217.73,111.20 209.54,128.11
             199.37,149.13 193.25,171.87 189.46,194.84
             189.46,194.84 186.70,216.24 186.70,216.24
             186.70,216.24 186.30,223.48 186.30,223.48
             186.30,223.48 185.70,232.29 185.70,232.29
             185.70,232.29 185.70,237.65 185.70,237.65
             185.70,237.65 185.39,242.68 185.39,242.68
             185.39,242.68 185.39,254.96 185.39,254.96
             185.39,254.96 185.70,259.68 185.70,259.68
             185.70,259.68 186.02,266.60 186.02,266.60
             186.02,266.60 186.70,276.36 186.70,276.36
             186.70,276.36 187.91,287.69 187.91,287.69
             189.77,286.90 191.36,285.37 192.94,284.10 Z
           M 180.98,391.34
           C 180.98,391.34 164.61,403.76 164.61,403.76
             147.88,417.59 131.50,431.92 116.14,447.28
             108.01,455.41 100.10,463.67 92.67,472.46
             79.14,488.48 68.38,505.47 58.78,524.08
             50.61,539.91 43.70,556.35 39.84,573.81
             38.48,579.98 37.54,586.09 37.11,592.38
             37.11,592.38 36.83,596.16 36.83,596.16
             36.83,596.16 36.83,604.97 36.83,604.97
             36.83,604.97 37.11,608.75 37.11,608.75
             37.96,620.84 40.55,633.87 44.71,645.26
             49.02,657.04 54.72,668.03 61.92,678.31
             71.63,692.19 83.92,704.56 97.26,714.96
             113.64,727.73 131.55,737.29 151.40,743.46
             162.04,746.77 173.05,749.02 184.13,750.13
             184.13,750.13 191.05,750.74 191.05,750.74
             191.05,750.74 194.20,751.02 194.20,751.02
             194.20,751.02 200.50,751.34 200.50,751.34
             200.50,751.34 207.74,751.34 207.74,751.34
             207.74,751.34 213.09,751.02 213.09,751.02
             220.98,751.01 228.89,749.88 236.69,748.73
             236.69,748.73 248.34,746.62 248.34,746.62
             248.34,746.62 245.85,731.51 245.85,731.51
             245.85,731.51 240.91,701.92 240.91,701.92
             240.91,701.92 225.86,609.69 225.86,609.69
             225.86,609.69 219.88,572.55 219.88,572.55
             219.88,572.55 217.18,555.56 217.18,555.56
             210.90,556.06 204.46,556.40 198.29,557.83
             179.92,562.10 164.62,571.69 154.56,587.98
             149.00,596.98 145.50,607.32 144.76,617.88
             144.76,617.88 144.47,621.66 144.47,621.66
             144.47,621.66 144.47,627.32 144.47,627.32
             144.48,632.12 145.49,638.75 146.81,643.37
             150.00,654.57 155.78,664.36 162.77,673.59
             167.48,679.81 172.50,685.09 178.78,689.76
             180.90,691.34 184.49,693.90 186.96,694.68
             184.88,696.48 182.96,698.01 181.93,700.66
             179.93,700.09 172.18,695.77 169.97,694.50
             160.74,689.21 152.01,683.02 144.47,675.48
             130.31,661.31 119.92,641.02 115.08,621.66
             113.70,616.15 111.80,605.80 111.74,600.25
             111.74,600.25 111.74,591.75 111.74,591.75
             111.74,591.75 112.28,585.77 112.28,585.77
             113.61,574.96 116.80,564.86 121.22,554.93
             133.31,527.78 155.68,504.71 182.24,491.46
             182.24,491.46 185.70,490.00 185.70,490.00
             190.80,487.80 195.16,486.22 200.50,484.69
             200.50,484.69 205.85,483.16 205.85,483.16
             205.85,483.16 202.20,461.13 202.20,461.13
             202.20,461.13 194.44,414.23 194.44,414.23
             194.44,414.23 189.80,385.90 189.80,385.90
             187.43,386.65 183.15,389.82 180.98,391.34 Z
           M 306.65,710.73
           C 311.03,704.67 314.81,697.62 317.37,690.59
             317.37,690.59 319.37,684.29 319.37,684.29
             321.85,676.66 323.04,667.42 323.59,659.43
             323.59,659.43 323.88,655.65 323.88,655.65
             323.88,655.65 323.88,641.80 323.88,641.80
             323.88,641.80 323.56,638.65 323.56,638.65
             323.56,638.65 323.56,635.82 323.56,635.82
             323.56,635.82 322.99,631.10 322.99,631.10
             320.82,609.46 314.08,588.71 297.75,573.52
             294.88,570.85 291.36,568.41 288.00,566.38
             277.19,559.86 267.04,557.10 254.63,555.47
             254.63,555.47 251.17,554.94 251.17,554.94
             251.17,554.94 248.34,554.94 248.34,554.94
             248.34,554.94 245.19,554.61 245.19,554.61
             245.19,554.61 240.78,554.35 240.78,554.35
             240.78,554.35 237.01,554.61 237.01,554.61
             237.01,554.61 231.03,554.61 231.03,554.61
             231.03,554.61 233.68,570.98 233.68,570.98
             233.68,570.98 239.03,604.34 239.03,604.34
             239.03,604.34 252.61,688.70 252.61,688.70
             252.61,688.70 257.96,722.06 257.96,722.06
             257.96,722.06 261.24,742.21 261.24,742.21
             278.99,736.06 295.48,726.18 306.65,710.73 Z
"""

################################################################################

bassClefVectorData = """
			M 401.96,0.42
			C 401.96,0.42 441.18,3.62 441.18,3.62
             529.77,14.21 620.86,49.11 684.24,113.79
             736.43,167.05 769.32,233.45 784.73,305.97
             788.61,324.25 793.89,352.30 794.11,370.69
             794.11,370.69 796.07,404.04 796.07,404.04
             797.22,502.49 755.63,595.84 699.43,674.70
             623.54,781.19 523.25,865.27 417.65,940.88
             417.65,940.88 317.65,1007.35 317.65,1007.35
             317.65,1007.35 249.02,1048.48 249.02,1048.48
             249.02,1048.48 101.96,1126.30 101.96,1126.30
             101.96,1126.30 52.94,1149.83 52.94,1149.83
             41.65,1155.44 32.77,1161.54 19.62,1157.74
             -2.68,1151.30 11.71,1130.56 22.13,1120.56
             26.50,1116.38 32.21,1112.84 37.25,1109.47
             37.25,1109.47 125.49,1051.23 125.49,1051.23
             125.49,1051.23 203.92,994.15 203.92,994.15
             235.35,970.57 265.32,944.78 294.12,918.08
             412.06,808.73 505.52,685.20 552.39,529.56
             567.44,479.59 581.00,419.03 580.38,366.77
             580.38,366.77 576.79,323.62 576.79,323.62
             565.87,221.50 531.34,111.50 427.45,69.53
             393.81,55.95 360.87,56.47 325.49,56.89
             325.49,56.89 290.20,60.64 290.20,60.64
             229.95,70.60 185.76,95.94 155.51,151.02
             147.57,165.48 140.35,180.65 134.67,196.13
             130.91,206.38 127.83,212.65 137.47,221.04
             144.96,227.57 153.65,226.47 162.75,225.00
             177.41,222.62 199.88,215.14 213.73,215.93
             259.23,218.51 304.66,235.73 335.03,270.69
             379.69,322.09 366.08,392.53 313.73,432.47
             280.86,457.54 238.72,467.27 198.04,466.79
             155.00,466.29 103.16,447.06 74.92,413.84
             66.18,403.56 57.77,385.22 52.39,372.66
             28.71,317.32 29.77,255.84 51.11,200.06
             83.59,115.16 168.52,45.85 252.94,16.37
             278.85,7.32 294.97,4.85 321.57,0.42
             321.57,0.42 401.96,0.42 401.96,0.42 Z
           M 986.69,149.06
           C 999.24,166.59 1000.24,179.52 999.99,200.06
             999.65,228.05 983.50,251.84 958.82,264.53
             943.54,272.39 934.27,272.82 917.65,272.62
             861.97,271.95 829.06,209.34 851.61,160.83
             865.29,131.40 889.10,120.09 919.61,116.72
             946.48,116.19 970.79,126.85 986.69,149.06 Z
           M 950.98,444.17
           C 1018.65,470.66 1016.62,569.00 947.06,591.62
             940.42,593.78 926.47,596.14 919.61,595.62
             904.95,594.50 888.52,589.91 876.47,581.27
             838.50,554.08 834.02,497.04 865.29,463.05
             876.44,450.93 892.01,443.78 907.84,440.25
             922.28,438.25 937.30,438.82 950.98,444.17 Z
"""

################################################################################


###############################################################################
# boxplots
###############################################################################

def myBoxPlot(arrDataRaw, arrLabels, arrAxis = [], title = '', fileName = '',
        plotMean = False, notch=0, sym='b+', vert=1, whis=[5, 95],
        positions=None, widths=None, hold=None, arrColors = None,
        ax = None, alpha = 1, logy = False
    ):

	# get rid of "None" values
	arrData = []
	for i in range(len(arrDataRaw)):
		tmp = []
		for val in arrDataRaw[i]:
			if not val is None:
				tmp.append(val)
		arrData.append(tmp)

	numBoxes = len(arrData)
	ax1 = ax
	if ax1 is None:
		fig = plt.figure()
		ax1 = fig.add_subplot(111)

	ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
	if logy: ax1.set_yscale('log')
	bp = ax1.boxplot(arrData, notch, sym, vert, whis, positions, widths, hold)
	#ax1.set_xticks(numpy.arange(numBoxes) + 1, arrLabels)
	ax1.set_xticklabels(arrLabels)
	pylab.setp(bp['boxes'], alpha=alpha)
	#pylab.xticks(numpy.arange(numBoxes) + 1, arrLabels)
	yMin = 0
	yMax = 0
	for i in range(numBoxes):
		if len(arrData[i]) > 0:
			mean = numpy.average(arrData[i])
			if i == 0:
				yMin = min(arrData[i])
				yMax = max(arrData[i])
			else:
				if yMin > min(arrData[i]):
					yMin = min(arrData[i])
				if yMax < max(arrData[i]):
					yMax = max(arrData[i])
			if plotMean == True:
				ax1.plot(i + 1, mean, 'ob')
				txt = 'mean = ' + str(mean)
				# ax1.text(0.6, 0.5, "test", size=50, rotation=30.,
				# 	ha="center", va="center",
				# 	bbox = dict(boxstyle="round",
				# 			 ec=(1., 0.5, 0.5),
				# 			 fc=(1., 0.8, 0.8),
				# 			 )
				# 	)

				#annotate (txt, xy=(i + 1.05, mean), weight='heavy', size=20,
				#	bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)), xycoords='data',
				#	xytext=(0, -45), textcoords='offset points',
				#)
	dataRange = yMax - yMin
	paddingTop = dataRange / 20.0
	if title != '':
		paddingTop += dataRange / 7.0
	ax1.axis([.5, numBoxes + .5, yMin - dataRange / 20.0, yMax + paddingTop], size=14)
	if title != '':
		ax1.annotate (title, xy = (0.8, yMax + dataRange/ 18.0), weight='heavy', size=16,
			bbox=dict(boxstyle="round", fc="0.8"))
	if len(arrAxis) == 2:
		ax1.xlabel(arrAxis[0])
		ax1.ylabel(arrAxis[1])

	plt.setp(bp['boxes'], color='black')
	plt.setp(bp['whiskers'], color='black')
	plt.setp(bp['fliers'], color='red', marker='+')

	if arrColors:
		medians = range(numBoxes)
		for i in range(numBoxes):
			box = bp['boxes'][i]
			boxX = []
			boxY = []
			for j in range(5):
				boxX.append(box.get_xdata()[j])
				boxY.append(box.get_ydata()[j])
				boxCoords = zip(boxX,boxY)

			k = i % len(arrColors)
			boxPolygon = plt.Polygon(boxCoords, facecolor=arrColors[k])
			ax1.add_patch(boxPolygon)

			# Now draw the median lines back over what we just filled in
			med = bp['medians'][i]
			medianX = []
			medianY = []
			for j in range(2):
				medianX.append(med.get_xdata()[j])
				medianY.append(med.get_ydata()[j])
				ax1.plot(medianX, medianY, 'k')
				medians[i] = medianY[0]
			# Finally, overplot the sample averages, with horixzontal alignment
			# in the center of each box
			ax1.plot([numpy.average(med.get_xdata())], \
				[numpy.average(arrData[i])], color='w', marker='*', \
				markeredgecolor='k', markersize=10)

	if fileName != '':
		plt.savefig(fileName)



###############################################################################
# violin plots
###############################################################################
def myViolinPlot(ax, arrData, arrLabels, vert=True, widths=0.5, showmeans=True,
        showextrema=True, showpercentiles=[50], points=100, bw_method=None,
        labelFontSize = 11, faceColor = 'yellow', edgeColor='darkblue',
		labelRotation = 0
    ):
	m = len(arrData)
	if len(arrLabels) != m:
		raise Exception("array size for data and labels doesn't match")
	arrTmp = []
	for i, data in enumerate(arrData):
		n = len(data)
		if n == 0:
			arrTmp.append([numpy.nan, numpy.nan])
		elif n == 1:
			arrTmp.append([data[0], numpy.nan])
		else:
			arrTmp.append(data)

	arrX = numpy.arange(m) + 0.5
	arrParts = None
	try:
		arrParts = ax.violinplot(arrTmp, arrX, vert=vert, showmeans=False, widths=widths,
		    showextrema=False, showmedians=False, points=points,
		    bw_method=bw_method)
		# ax.violinplot(arrTmp, arrX, vert=vert, widths=widths, showmeans=showmeans,
         #    showextrema=showextrema, showmedians=showmedians, points=points, bw_method=bw_method,
         #    hold=hold, data=data)
		#ax.violinplot(arrTmp, arrX)
	except Exception as e:
		print "WARNING: %s" % (str(e))
	if showextrema:
		for idx, data in enumerate(arrTmp):
			x = idx + 0.5
			valMin = numpy.nanmin(data)
			valMax = numpy.nanmax(data)
			for y in [valMin, valMax]:
				ax.plot([x - widths/2.0, x + widths/2.0], [y, y], '-',
				    linewidth=0.5, color='black', alpha=0.6)
			ax.plot([x, x], [valMin, valMax], '-',
				    linewidth=0.5, color='black', alpha=0.6)
	if isinstance(showpercentiles, list):
		for perc in showpercentiles:
			for idx, data in enumerate(arrTmp):
				x = idx + 0.5
				y = numpy.percentile(data, perc)
				ax.plot([x - widths/4.0, x + widths/4.0], [y, y], '-',
				    linewidth=1.4, color='red')
	if showmeans:
		for idx, data in enumerate(arrTmp):
			y = numpy.nanmean(data)
			ax.plot(idx + 0.5, y, '*', color='lightblue', markersize=8)
	ax.set_xticks(arrX)
	ax.set_xticklabels(arrLabels, size=labelFontSize, rotation = labelRotation)
	ax.set_xlim(arrX[0] - 0.5, arrX[-1] + 0.5)
	if arrParts:
		for pc in arrParts['bodies']:
			pc.set_color(faceColor)
			pc.set_edgecolor(edgeColor)



###############################################################################
# scatter plots
###############################################################################

def myScatterPlot(arrData, arrLabels, arrAxis = [], title = '', fileName = '',
        arrColors = None, arrMarkers = None, arrMarkersize = None
    ):
	print title
	fig = plt.figure()


	gs = gridspec.GridSpec(1, 12)
	gs.update(left=0.1, right=0.95, wspace=0.7, bottom = 0.12)
	#gs.update(left=0.05, right=0.48, wspace=0.05)
	#["left", "bottom", "right", "top", "wspace", "hspace"]
	ax1 = plt.subplot(gs[0, :-2])

	#ax1 = fig.add_subplot(111)

	ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
	ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

	arrColorDefault = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	arrPlots = []
	xMin = 0
	xMax = 0
	yMin = 0
	yMax = 0
	arrRegLines = []
	dataX = []
	dataY = []
	for i in range(len(arrData)):
		marker = "o"
		color = arrColorDefault[i % len(arrColorDefault)]
		markerSize = 5
		if arrColors: color = arrColors[i % len(arrColors)]
		if arrMarkersize: markerSize = arrMarkersize[i % len(arrMarkersize)]
		if arrMarkers: marker = arrMarkers[i % len(arrMarkers)]
		p = plt.plot(arrData[i][0], arrData[i][1], marker, color=color, markersize = markerSize)
		arrPlots.append(p)
		xMaxLocal = max(arrData[i][0])
		xMinLocal = min(arrData[i][0])
		if i == 0:
			xMin = xMinLocal
			xMax = xMaxLocal
		else:
			if xMinLocal < xMin: xMin = xMinLocal
			if xMaxLocal > xMax: xMax = xMaxLocal
		yMaxLocal = max(arrData[i][1])
		yMinLocal = min(arrData[i][1])
		if i == 0:
			yMin = yMinLocal
			yMax = yMaxLocal
		else:
			if yMinLocal < yMin: yMin = yMinLocal
			if yMaxLocal > yMax: yMax = yMaxLocal
		dataX += arrData[i][0]
		dataY += arrData[i][1]

		# regression line
		gradient, intercept, r_value, p_value, std_err = stats.linregress(arrData[i][0], arrData[i][1])
		print arrLabels[i], "... gradient:", gradient, "... intercept:",intercept, "... R squared:", r_value ** 2
		style = '--'
		plt.plot([xMinLocal, xMaxLocal], [intercept + gradient * xMinLocal, intercept + gradient * xMaxLocal], style, color=color, linewidth = 0.7)
		txtReg = 'y = ' + ('%0.3f' % gradient) + 'x + ' + ('%0.3f' % intercept) + '; R2 = ' + ('%0.3f' % (r_value ** 2.0))
		arrRegLines.append(txtReg)


	rangeX = xMax - xMin
	rangeY = yMax - yMin
	for i in range(len(arrRegLines)):
		plt.text(xMin, yMax - (rangeY / 100.0 + i * rangeY / 20.0), arrRegLines[i], color = arrColors[i], fontsize = 8)
	gradient, intercept, r_value, p_value, std_err = stats.linregress(dataX, dataY)
	print "OVERALL: ... gradient:", gradient, "... intercept:",intercept, "... R squared:", r_value ** 2
	style = '--'
	plt.plot([xMin, xMax], [intercept + gradient * xMin, intercept + gradient * xMax], style, color="red", linewidth = 1.7)
	txtReg = 'y = ' + ('%0.3f' % gradient) + 'x + ' + ('%0.3f' % intercept) + '; R2 = ' + ('%0.3f' % (r_value ** 2.0))
	plt.text(xMin, yMax - (rangeY / 100.0 + (len(arrRegLines)) * rangeY / 20.0), txtReg, color = "red", fontsize = 8)


	factor = 0.1
	ax1.set_xlim(xMin - rangeX * factor, xMax + rangeX * factor)
	ax1.set_ylim(yMin - rangeY * factor, yMax + rangeY * factor)

	if title != '':
		#plt.annotate (title, xy = (0.8, yMax + dataRange/ 18.0), weight='heavy', size=16,
		#	bbox=dict(boxstyle="round", fc="0.8"))
		plt.title(title)
	if len(arrAxis) == 2:
		plt.xlabel(arrAxis[0])
		plt.ylabel(arrAxis[1])

	#plt.legend(arrPlots, arrLabels, loc=2)
	plt.legend(arrPlots, arrLabels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.subplots_adjust(hspace=0.4)

	if fileName != '':
		plt.savefig(fileName)
	else:
		plt.show()





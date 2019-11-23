"""

@package generalUtility This module contains some general-purpose utility functions

@copyright GNU Public License
@author written 2009-2014 by Christian Herbst (www.christian-herbst.org) 
@author Partially supported by the SOMACCA advanced ERC grant, University of Vienna, 
	Dept. of Cognitive Biology

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

@warning DISCLAIMER: this module (and the others in this library) was developed 
on a Mac, and was never really tested a Windows platform. There might be 
problems with the backslashes used in Windows path indicators.
"""

import numpy, math, cPickle, os, string, sys, datetime, time, random, operator, copy


######################################################################

def myMkDir(newdir):  
	"""
	http://code.activestate.com/recipes/82465-a-friendly-mkdir/
	works the way a good mkdir should :)
		- if already exists, silently complete
		- regular file in the way, raise an exception
		- parent directory(ies) does not exist, make them as well
	@param newdir the full path name of the directory to be created
	"""
	if os.path.isdir(newdir):
		pass
	elif os.path.isfile(newdir):
		raise OSError("a file with the same name as the desired " \
					  "dir, '%s', already exists." % newdir)
	else:
		head, tail = os.path.split(newdir)
		if head and not os.path.isdir(head):
			myMkDir(head)
		#print "_mkdir %s" % repr(newdir)
		if tail:
			os.mkdir(newdir)

######################################################################

def saveData(data, fileName):
	"""
	saves any data structure as a Python pickle. saved data can be retrieved
	by calling @ref loadData
	@param fileName the name of the file that is being created
	"""
	if sys.version_info[0] >= 3:
		f = open(fileName, "wb")
	else:
		f = open(fileName, "w")
	if f:
		cPickle.dump(data, f)
		f.close()
	else:
		raise Exception("unable to save data to file '" \
			+ fileName + "'")

######################################################################

def loadData(fileName):
	"""
	loads any data structure that was previously saved as a Python pickel with
	@ref saveData
	@param fileName the name of the file that contains the data structure
	"""
	f = None
	if sys.version_info[0] >= 3:
		f = open(fileName, "rb")
	else:
		f = open(fileName, "r")
	if f:
		data = None
		if sys.version_info[0] >= 3:
			data = cPickle.load(f, encoding='latin1')
		else:
			data = cPickle.load(f)
		f.close()
		return data
	else:
		msg = "unable to load parameters file '" + fileName + "'"
		raise Exception(msg)
	
######################################################################

def sanitizePath(path, failIfEmptyString = True):
	"""
	converts all backslashes to forward slashes and adds a slash at
	the end of the given string, if not already present
	@param path the path that should be sanitized
	@return returns the sanitized path
	"""
	if path == '' or path is None: 
		if failIfEmptyString:
			raise Exception("path must not be empty")
		else:
			return '/'
	path.replace("\\", "/")
	if path[-1] != '/': path += '/'
	return path

######################################################################

def getCurrentExecutionPath():
	"""
	returns the path of the Python script that is currently being 
	executed
	"""
	#return sanitizePath(sys.path[0])
	path = os.getcwd()
	return sanitizePath(path)
	
######################################################################

def getXofMax(data):
	"""
	locates the index of the maximum value found in a list or an array
	@param data the list or array that should be analyzed
	@return the index position (zero-based) of the maximum, or None if no
		maximum is found
	"""
	if len(data) < 1:
		raise Exception("list is empty")
	valMax = data[0]
	xOfMax = 0
	for i in range(len(data)):
		if data[i] > valMax:
			valMax = data[i]
			xOfMax = i
	return xOfMax

######################################################################

def findArrayMaximum(
		data, 
		offsetLeft = 0, 
		offsetRight = -1, # if -1, the array size will be used
		doInterpolate = True, # increase accuracy by performing a 
							  # parabolic interpolation
):
	"""
	@param data a numpy array
	@param offsetLeft the index position at which analysis will commence
	@param offsetRight the terminating index position. if -1, the array size 
		will be used
	@param doInterpolate if True: increase accuracy by performing a 
		parabolic interpolation within the results
	@return a list containing the index and the value of the maximum
	"""
	objType = type(data).__name__.strip()
	if objType != "ndarray":
		raise Exception('data argument is no instance of numpy.array')
	size = len(data)
	if (size < 1):
		raise Exception('data array is empty')
	xOfMax = -1
	valMax = min(data)
	if offsetRight == -1:
		offsetRight = size
	for i in range(offsetLeft + 1, offsetRight - 1):
		if data[i] >= data[i-1] and data[i] >= data[i + 1]:
			if data[i] > valMax:
				valMax = data[i]
				xOfMax = i
	if doInterpolate:
		if xOfMax > 0 and xOfMax < size - 1:
			# use parabolic interpolation to increase accuracty of result
			alpha = data[xOfMax - 1]
			beta = data[xOfMax]
			gamma = data[xOfMax + 1]
			xTmp = (alpha - gamma) / (alpha - beta * 2 + gamma) / 2.0
			xOfMax = xTmp + xOfMax
			valMax = interpolateParabolic(alpha, beta, gamma, xTmp)
	if xOfMax == -1:
		raise Exception("no maximum found")
	return [xOfMax, valMax]
	
######################################################################
	
def findPeaks(
		data, 
		offsetLeft = 0, 
		offsetRight = -1, # if -1, the array size will be used
		doSortResults = True, # sort peaks (y-value) in descending order
		doInterpolate = True, # increase accuracy by performing a 
							  # parabolic interpolation
		threshold = None, # if given (a float), we'll only consider peaks
						  # greater or equal that threshold
):
	"""
	finds all the the peaks (i.e., the maxima) in the provided data array
	@param data a numpy array
	@param offsetLeft  
	@param offsetRight if -1, the array size will be used
	@param doSortResults if True: sort peaks (y-value) in descending order
	@param doInterpolate if True: increase accuracy by performing a parabolic
		interpolation
	@return a list containing lists of the index positions and the values of the
		found maxima
	"""
	objType = type(data).__name__.strip()
	if objType != "ndarray":
		raise Exception('data argument is no instance of numpy.array')
	size = len(data)
	if (size < 1):
		raise Exception('data array is empty')
	arrPeaksX = []
	arrPeaksY = []
	if offsetRight == -1:
		offsetRight = size
	for i in range(offsetLeft + 1, offsetRight - 1):
		found = False
		if data[i] >= data[i-1] and data[i] >= data[i + 1]:
			found = True
			if not threshold is None:
				if data[i] < threshold:
					found = False
		if found:
			xOfMax = i
			valMax = data[i]
			if doInterpolate:
				if xOfMax > 0 and xOfMax < size - 1:
					# use parabolic interpolation to increase accuracty of result
					alpha = data[xOfMax - 1]
					beta = data[xOfMax]
					gamma = data[xOfMax + 1]
					xTmp = (alpha - gamma) / (alpha - beta * 2 + gamma) / 2.0
					xOfMax = xTmp + xOfMax
					valMax = interpolateParabolic(alpha, beta, gamma, xTmp)
			arrPeaksX.append(xOfMax)
			arrPeaksY.append(valMax)
	# sort results
	if doSortResults:
		doIt = True
		while(doIt):
			doIt = False
			for i in range(len(arrPeaksY) - 1):
				if arrPeaksY[i] < arrPeaksY[i + 1]:
					doIt = True
					tmp = arrPeaksY[i + 1]
					arrPeaksY[i + 1] = arrPeaksY[i]
					arrPeaksY[i] = tmp
					tmp = arrPeaksX[i + 1]
					arrPeaksX[i + 1] = arrPeaksX[i]
					arrPeaksX[i] = tmp
					break
	return arrPeaksX, arrPeaksY
	
######################################################################

def interpolateLinear(
		y1, #
		y2, #
		x # weighting [0..1]. 0 would be 100 % y1, 1 would be 100 % y2
):
	""" 
	simple linear interpolation between two variables 
	@param y1 
	@param y2
	@param x weighting [0..1]: 0 would be 100 % y1, 1 would be 100 % y2
	@return the interpolated value
	"""
	return y1 * (1.0 - x) + y2 * x

######################################################################

def interpolateParabolic(
		alpha, 
		beta, 
		gamma, 
		x # relative position of read offset [-1..1]
):
	""" 
	parabolic interpolation between three equally spaced values
	@param alpha first value
	@param beta second value
	@param gamma third value
	@param x relative position of read offset [-1..1]
	@return the interpolated value
	"""
	if (x == 0): return beta
	
	#we want all numbers above zero ...
	offset = alpha;
	if (beta < offset): offset = beta
	if (gamma < offset): offset = gamma
	offset = math.fabs(offset) + 1
	
	alpha += offset;
	beta += offset;
	gamma += offset;
	
	a = b = c = 0;
	a = (alpha - 2.0 * beta + gamma) / 2.0
	if (a == 0):
		if (x > 1):
			return interpolateLinear(beta, gamma, x) - offset
		else:
			return interpolateLinear(alpha, beta, x + 1) - offset
	else:
		c = (alpha - gamma) / (4.0 * alpha)
		b = beta - a * c * c
		return (a * (x - c) * (x - c) + b) - offset

######################################################################

def removeNanInf(arrData, arrData2 = None):
	"""
	Removes all NaN and Inf data points from the given input array(s).
	If two matched arrays are provided, both respective data values are 
	being removed, even if only one of the values is NaN or Inf
	@param arrData: a 1D numpy array or a list containing floats
	@param arrData2: None, or a 1D numpy array or a list containing 
		floats. If not None, both arrays are treated as linked (matched)
		data
	@return one or two 1D numpy arrays containing the cleaned data
	"""
	arrDataOut = []
	arrDataOut2 = []
	
	numArrays = 1
	if not arrData2 is None: numArrays = 2
	
	n = len(arrData)
	if numArrays == 2:
		m = len(arrData2)
		if n != m: 
			raise Exception("if two arrays are provided, they need to have the same length")
			
	for i in range(n):
		val = arrData[i]
		if (not numpy.isnan(val)) and (not numpy.isinf(val)): 
			if numArrays == 2:
				val2 = arrData2[i]
				if (not numpy.isnan(val2)) and (not numpy.isinf(val2)): 
					arrDataOut.append(val)	
					arrDataOut2.append(val2)	
			else:
				arrDataOut.append(val)			
	
	arrDataOut = numpy.array(arrDataOut, numpy.float32)
	if numArrays == 2:
		arrDataOut2 = numpy.array(arrDataOut2, numpy.float32)
		return arrDataOut, arrDataOut2
	return arrDataOut

######################################################################
	
def splitFullFileName(fileName):
	"""
	split a full file name into path, fileName and suffix
	@param fileName
	@return a list containing the path (with a trailing slash added), the 
		file name (without the suffix) and the file suffix (without the 
		preceding dot)
	"""
	tmp = fileName.split('/')
	path = '/'.join(tmp[:-1]) + '/'
	fullFileName = tmp[-1]
	tmp2 = fullFileName.split('.')
	fileName = '.'.join(tmp2[:-1])
	suffix = tmp2[-1]
	return path, fileName, suffix

######################################################################

def getFileNameOnly(fileName):
	"""
	return the file name minus the trailing suffix
	"""
	return '.'.join(fileName.split('/')[-1].split('.')[:-1])

######################################################################

def toDerivative(
		data, 
		derivativeType = 2,
		normalize = -1
	):
	"""
	@deprecated function moved to @ref dspUtil
	"""
	raise Exception("toDerivative(...) has been moved to the module dspUtil")
	
######################################################################

def calculateLinearRegressionFit(data):
	""" 
	fit a linear line on the input data. input data must be an array containing 
	two arrays, one ([0]) for the x-axis data and one ([1]) for the y-axis data.
	"""
	a = 0
	b = 0
	dev = 0
	try:
		if len(data) != 2:
			raise Exception("calculateLinearRegressionFit -need 2D vector array")
		if len(data[0]) != len(data[1]):
			raise Exception("calculateLinearRegressionFit - size of input arrays does not match")
		if len(data[0]) == 0:
			raise Exception("calculateLinearRegressionFit - input array empty")
	except:
		raise Exception("calculateLinearRegressionFit - data structure invalid")

	sumx = 0
	sumy = 0
	sumxx = 0
	sumyy = 0
	sumxy = 0

	n = len(data[0])
	for i in range(n):
		x = data[0][i]
		y = data[1][i]
		sumx += x
		sumy += y
		sumxx += x * x
		sumyy += y * y
		sumxy += x * y

	n = float(n)
	Sxx = sumxx - sumx * sumx / n
	Sxy = sumxy - sumx * sumy / n
	b = Sxy / Sxx
	a = (sumy - b * sumx) / n

	dev = 0
	for i in range(n):
		currentResidual = data[1][i] - (a + b * data[0][i]);
		dev += currentResidual
	dev /= n
	return a, b, dev

######################################################################	

def makeSystemCall(args, shell=False):
	""" 
	make a system call 
	@param args must be an array, the first entry being the called program 
	@shell turns on or off shell based features. Warning: using shell=True
		might be a potential security hazard.
	@return returns a tuple with communication from the called system process, 
		consisting of stdoutdata, stderrdata
	"""
	import subprocess
	proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
	msg = proc.communicate()
	try:
		proc.kill()
	except:
		pass
	#msg = subprocess.call(args) - recommended version; we don't use it, since we want to get back the system message
	return msg

##############################################################################
	

def intToRoman(valInt):
	"""
	convert an integer to a roman numeral
	taken from http://code.activestate.com/recipes/81611-roman-numerals/
	@param valInt the integer that should be converted into a romal numeral
	@return a string containing the roman numeral
	"""
	if type(valInt) != type(1):
		raise TypeError("expected integer, got %s" % type(valInt))
	if not 0 < valInt < 4000:
		raise ValueError("Argument must be between 1 and 3999")
	ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
	nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
	result = ""
	for i in range(len(ints)):
		count = int(valInt / ints[i])
		result += nums[i] * count
		valInt -= ints[i] * count
	return result	
	
##############################################################################

def getUserTmpDir():
	"""
	looks for the users home dir, appends tmp/ to that path. If the resulting
	path does not exist it is silently being created.
	@return the (new) user tmp dir
	"""
	from os.path import expanduser
	tmpDataPath = expanduser("~") + '/tmp/'
	myMkDir(tmpDataPath)
	return tmpDataPath
	
##############################################################################

def getCurrentTimestamp(format = None):
	"""
	returns a formatted timestamp
	@param format if None, we'll return the milliseconds since the unix epoch
	       as a number
	"""
	if format is None:
		return time.time() * 1000
	# format = "%Y-%m-%d %H:%M:%S:%f"
	now = datetime.datetime.now()
	return now.strftime(format)

##############################################################################
			
def getRandomSequence(n, asciiOnly = False):
	"""
	returns a random sequence of characters
	@param n the length of the returned sequence
	"""
	if n < 1 or n > 128:
		raise Exception("n out of range")
	if not asciiOnly:
		try:
			return os.urandom(n)
		except:
			pass
	random.seed()
	txt = ''
	for i in range(n):
		txt += chr(40 + int(round(random.random() * 87)))
	return txt
	

##############################################################################

class CTimer:

	def __init__(self):
		self.reset()

	def reset(self):
		self.__startTime = time.time()

	def get(self):
		return time.time() - self.__startTime

	def getStartTime(self):
		return self.__startTime

######################################################################


TYPE_INT = 1
TYPE_FLOAT = 2
TYPE_STRING = 3
TYPE_BOOL = 4
ARR_TYPES = {
	TYPE_INT: 'int',
	TYPE_FLOAT: 'float',
	TYPE_STRING: 'string',
	TYPE_BOOL: 'bool',
}

class CParams:

	"""
	a class that can store any number of parameters of various types.
	"""

	def __init__(self):
		self.__arrParams = {}

	def get(self, p, failIfNotFound = True, defaultValue = None, ignoreCase = True):
		"""
		retrieve a parameter's value
		:param p: the parameter, indicated as a string
		:param failIfNotFound: throw an exception if the parameter is not found
			and not default value is given
		:param defaultValue: if not None, this value is returned if the parameter
			is not found
		:return:
		"""
		for p2 in self.__arrParams:
			if p.upper() == p2.upper():
				return self.__arrParams[p2]

		if not defaultValue is None:
			return defaultValue
		else:
			if failIfNotFound:
				raise Exception("parameter '%s' not found" % str(p))
			else:
				return None

	def set(self, p, val, pType):
		if not pType in ARR_TYPES:
			raise Exception("parameter type '%s' not recognized" % str(pType))
		if pType == TYPE_INT:
			if not isinstance(val, int):
				raise Exception("value must be an integer")
		elif pType == TYPE_FLOAT:
			if not type(val).__name__ in ['int', 'float', 'double', 'float32', 'float64']:
				#if not isinstance(val, float):
				raise Exception("value must be a float")
		elif pType == TYPE_STRING:
			if not type(val).__name__ in ['str', 'unicode']:
				raise Exception("value must be a string")
		elif pType == TYPE_BOOL:
			if not isinstance(val, bool):
				raise Exception("value must be boolean")
		if not type(p).__name__ in ['str', 'unicode']:
			raise Exception("parameter name p must be a string")
		self.__arrParams[p] = val

	def setInt(self, p, val):
		self.set(p, val, TYPE_INT)
	def setFloat(self, p, val):
		self.set(p, val, TYPE_FLOAT)
	def setString(self, p, val):
		self.set(p, val, TYPE_STRING)
	def setBool(self, p, val):
		self.set(p, val, TYPE_BOOL)

	def listParams(self):
		"""
		get params as text with newlines
		@return:
		"""
		txt = 'CParams:'
		for p in self.__arrParams:
			txt += "\t%s: %s\n" % (str(p), str(self.__arrParams[p]))
		return txt

	def __str__(self):
		n = len(self.__arrParams)
		if n > 5:
			return "CParams (%d entries)" % n
		return self.listParams()


######################################################################
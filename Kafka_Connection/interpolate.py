def interpolate_color(a, b):
	# a: color of (h, s, v)
	# b: color of (h, s, v)
	h1, s1, v1 = a
	h2, s2, v2 = b

	s3 = int((s1 + s2)/2)
	v3 = int((v1 + v2)/2)
	h1, h2 = min(h1, h2), max(h1, h2)
	if h2 - h1 < 360 + h1 - h2:
		h3 = int((h1 + h2)/2)
	else:
		h3 = int((360 + h1 + h2)/2)

	return h3, s3, v3

if __name__ == '__main__':
	red = (0, 67, 15)
	violet = (313, 80, 31)
	yellow = (62, 31, 8)

	print('red and violet: ', interpolate_color(red, violet))
	print('red and yellow: ', interpolate_color(red, yellow))

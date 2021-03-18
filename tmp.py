import os
os.chdir("processed")
directories = []
for dir in os.listdir():
	underscore = ''
	for i, char in enumerate(dir):
		if char == '_':
			underscore = i
	if dir[:underscore] not in directories:
		directories.append(dir[:underscore])
print(directories)
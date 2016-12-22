import glob

year = '2015'

def replace_protected(fname, year):
    repl_str = 'http://bebi103.caltech.edu/' + str(year) \
                    + '/protected/'
    with open(fname, 'r') as f:
        contents = f.read()
        contents = contents.replace('protected/', repl_str)
    with open(fname, 'w') as f:
        f.write(contents)

for fname in glob.iglob('*.html'):
    replace_protected(fname, year)

for fname in glob.iglob('*/**.html', recursive=True):
    replace_protected(fname, year)

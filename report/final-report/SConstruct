import os

env = Environment(
    tools=['pdflatex', 'pdftex', 'tex'],
    )

env = Environment( 
    tools = [ 'pdflatex' ] ,
    ENV = {'PATH' : os.environ['PATH']}
    )

env['SHELL'] = env.WhereIs('bash')

# Look in standard directory ~/texmf for .sty files
env['ENV']['TEXMFHOME'] = os.path.join(os.environ['HOME'],'texmf')

env.AppendUnique(
    #PDFLATEXFLAGS=['-file-line-error', '-interaction=batchmode'],
    #BIBTEXFLAGS='-terse',
    )

env.PrependENVPath('PATH', [
	'/s/texlive-2011/bin',
	'/s/texlive-2010/bin',
	])

pdf = env.PDF('main.tex')[0]
Default(pdf)

ps = env.Command('main.ps', pdf, 'pdftops $SOURCE $TARGET')

Depends(pdf, [
	 Glob('section/*.tex'),
#        Glob('figures/*.c'),
#        Glob('figures/why-merge/*.txt'),
#        Glob('figures/traps/*.c'),
#        Glob('figures/traps/*.pdf'),
#        Glob('figures/traps/*.tex'),
        'jpaper.cls',
        ])

@main.group(invoke_without_command=False,
            help="Работа с классификационными моделями моделями")
def models():
  print()
  
@models.command(help="Установка моделей из онлайна")
@click.option('-m', '--model',
              type=str)
def install(model):
  print(model)
  
@models.command(help="Вывод списка моделей")
@click.option('-o', '--online',
              default=False,
              is_flag=True,
              help="Вывод моделей, доступных онлайн")
def list(online):
  if online:
    print("Данный функционал в процессе реализации")
  else:
    pass
  
class Model(click.MultiCommand):

    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(plugin_folder):
            if filename.endswith('.py') and filename != '__init__.py':
                rv.append(filename[:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        ns = {}
        fn = os.path.join(plugin_folder, name + '.py')
        with open(fn) as f:
            code = compile(f.read(), fn, 'exec')
            eval(code, ns, ns)
        return ns['cli']

cli = MyCLI(help='This tool\'s subcommands are loaded from a '
            'plugin folder dynamically.')
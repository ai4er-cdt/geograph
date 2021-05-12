""" Module with utility methods for docs"""


import glob

def create_notebook_links():
    """Create links and entries for notebook in sphinx docs.

    Example usage (in home repo directory):
    > import docs.utils
    > docs.utils.create_notebook_links()
    """
    notebooks = glob.glob("./notebooks/*.ipynb")
    notebooks.sort()

    file_name_tmp = 'docs/notebooks/{}.nblink'
    file_content_tmp = """
{{
"path": "../../notebooks/{}.ipynb"
}}
"""

    rst_index = """
Advanced Tutorials
======================

.. toctree::
   :maxdepth: 1
    """

    for path in notebooks:
        nb_name = path.split("/")[-1].split(".")[0]
        file_name = file_name_tmp.format(nb_name)
        file_content = file_content_tmp.format(nb_name)
        with open(file_name,'w') as f:
                f.write(file_content)
                rst_index += "\n   notebooks/{}".format(nb_name)

    with open("docs/tutorials.rst",'w') as f:
        f.write(rst_index)

if __name__ == "__main__":
    create_notebook_links()
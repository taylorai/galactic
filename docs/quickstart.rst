Quick Start
=================

Setup dataset
-------------------
Once you have installed Galactic, you can start using it by following the steps below:

1. Import GalacticDataset

.. code:: python 

   from galactic import GalacticDataset


2. Load a Dataset

.. note::
      See :doc:`GalacticDataset.loaders <loaders>` for more options on loading datasets (e.g. parquet, HuggingFace, jsonl, pandas, etc.)

.. code:: python 

   # Load from a csv file
   ds = GalacticDataset.from_csv("path/to/dataset")

3. Verify the Dataset

.. code:: python

   # Confirm the dataset loaded
   print(ds)

   # Verify the first element of the dataset
   first_element = ds[0]
   print(first_element)

   # Verify the column names and length of the dataset
   print(ds.column_names, len(ds))

Optional Parameters
-------------------

Set OpenAI API Key


.. code:: python
   
   # Set your OpenAI API key
   import os

   ds.set_openai_key(os.environ["OPENAI_API_KEY"],)


Set Rate Limits

.. code:: python

   # Adjust the rate limits as per your requirements
   ds.set_rate_limits(
    max_tokens_per_minute=350_000,
    max_requests_per_minute=4_000
   )

Additional Methods
-------------------
GalacticDataset class also includes various methods that allow you to load data, filter data, tag data, transform data, classify data, and more. Check out the GalacticDataset methods via the sidebar navigation.

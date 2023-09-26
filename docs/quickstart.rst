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
   first_element = galactic_dataset[0]
   print(first_element)

Optional Parameters
-------------------

Set OpenAI API Key


.. code:: python
   
   galactic_dataset.set_openai_key("your-api-key")


Set Rate Limits

.. code:: python

   galactic_dataset.set_rate_limits(max_tokens_per_minute=10000, max_requests_per_minute=100)

Additional Methods
-------------------
GalacticDataset class also includes various methods that allow you to load data, filter data, tag data, transform data, classify data, and more, as per the methods attached to it in the shared code.

Notes
-------------------
Make sure to replace "your-api-key" with your actual OpenAI API key.
Adjust the rate limits as per your requirements.
This is a basic quick-start guide, and you can add more details and refinements as per the specific functionalities and features provided by your GalacticDataset class. Also, remember to provide clear and concise documentation for each method in the class to help users understand how to use them effectively.
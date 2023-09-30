conversations
=============
Functions for processing conversation data may require you to provide a MessageConfig (which can either be a MessageConfig object from galactic.utils, or just a dictionary with the same fields). It it, you just need to specify the following fields:

- role_field (str): What field in a message specifies the speaker? For OpenAI chats, it's "role" In ShareGPT, it's "from". etc.
- content_field (str): What field in a message specifies the content? For OpenAI chats, it's "content". It may also be "value", "text", etc. in other formats.
- user_role (str): In the role field, what value refers to the human user? It may be "user" (for OpenAI), but other formats may have things like "human".
    assistant_role (str): In the role field, what value refers to the assistant? It may be "assistant" (for OpenAI), but other formats may have things like "bot", "gpt", etc.
    system_role (str): In the role field, what value refers to the system? It may be "system" (for OpenAI), or some formats may not use system prompts (in this case, it's fine to just use "system").


.. automodule:: galactic.conversations
   :members:

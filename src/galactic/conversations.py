import re
import os
from typing import Optional, Union, Literal
from .utils import MessageConfig, OPENAI_MESSAGE_CONFIG


def conversation_from_dicts(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    output_column: Optional[str] = None,
    output_message_config: MessageConfig = OPENAI_MESSAGE_CONFIG,
    conversation_field: Optional[str] = None,
    metadata_fields: Optional[list[str]] = [],
):
    """
    Parses (somewhat) arbitrary JSON containing conversations into standard format for OpenAI/etc. conversational fine-tuning.
    If output column is not provided, overwrites the input column.
    """
    if output_column is None:
        output_column = input_column
    if isinstance(input_message_config, dict):
        input_message_config = MessageConfig(**input_message_config)
    role_dict = {
        input_message_config.user_role: output_message_config.user_role,
        input_message_config.assistant_role: output_message_config.assistant_role,
        input_message_config.system_role: output_message_config.system_role,
    }

    def _parse_messages(sample: dict):
        conversation = (
            sample[input_column]
            if conversation_field is None
            else sample[input_column][conversation_field]
        )
        conversation = list(
            map(
                lambda msg: {
                    output_message_config.role_field: role_dict[
                        msg[input_message_config.role_field]
                    ],
                    output_message_config.content_field: msg[
                        input_message_config.content_field
                    ],
                    "metadata": {k: msg[k] for k in metadata_fields},
                },
                conversation,
            )
        )
        return {output_column: conversation}

    self.dataset = self.dataset.map(_parse_messages)


def conversation_from_string(
    self,
    input_column: str,
    user_delimiter: str,
    assistant_delimiter: str,
    system_delimiter: str,
    output_column: Optional[str] = None,
    output_message_config: MessageConfig = OPENAI_MESSAGE_CONFIG,
    strip_whitespace: bool = True,
    replace_values: Optional[dict[str, str]] = None,
):
    """
    Parses a conversation from a string 'transcript' into a structured object with a list of messages.
    """
    if output_column is None:
        output_column = input_column

    pattern = f"({re.escape(user_delimiter)}|{re.escape(assistant_delimiter)}|{re.escape(system_delimiter)})"

    def _parse_single(s: str):
        tokens = re.split(pattern, s)[1:]  # Skip the initial empty string
        result = []
        for i in range(0, len(tokens), 2):
            delimiter, content = tokens[i], tokens[i + 1]
            role = {
                user_delimiter: output_message_config.user_role,
                assistant_delimiter: output_message_config.assistant_role,
                system_delimiter: output_message_config.system_role,
            }.get(delimiter, "unknown")

            if replace_values is not None:
                for k, v in replace_values.items():
                    content = content.replace(k, v)
            if strip_whitespace:
                content = content.strip()

            result.append({"content": content, "role": role})

        return result

    self.dataset = self.dataset.map(
        lambda x: {output_column: _parse_single(x[input_column])}
    )
    return self


def convert_conversation_to_string(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    output_column: str,
    user_delimiter: str,
    assistant_delimiter: str,
    system_delimiter: str,
    strip_whitespace: bool = True,
):
    """
    Converts a conversation from the standard format to a string, using the provided delimiters (e.g. User:, Assistant:)
    """
    if isinstance(input_message_config, dict):
        input_message_config = MessageConfig(**input_message_config)

    def _convert_single(conversation: list[dict]):
        result = ""
        for msg in conversation:
            role, content = (
                msg[input_message_config.role_field],
                msg[input_message_config.content_field],
            )
            result += {
                input_message_config.user_role: user_delimiter,
                input_message_config.assistant_role: assistant_delimiter,
                input_message_config.system_role: system_delimiter,
            }.get(role, "unknown") + content
        if strip_whitespace:
            result = result.strip()
        return result

    self.dataset = self.dataset.map(
        lambda x: {output_column: _convert_single(x[input_column])}
    )
    return self


def get_conversation_length(self, input_column: str, output_column: str):
    """
    Get the length (in messages) of a conversation. If you want length in tokens, convert it to a string, and then use count_tokens
    from the taggers module.
    """
    self.dataset.map(lambda x: {output_column: len(x[input_column])})
    return self


def get_conversation_speakers(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    output_column: str = "__speakers",
):
    """
    Computes list of just the speakers for each message in each conversation.
    """
    self.dataset.map(
        lambda x: {
            output_column: [
                msg[input_message_config.role_field] for msg in x[input_column]
            ]
        }
    )
    return self


def is_alternating(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    output_column: str = "__alternating",
):
    """
    Returns true if no speaker goes 2x in a row. Useful if you make assumptions about conversations being alternating
    and want to catch ones where they're not.
    """

    def _check_alternating(conversation: list[dict]):
        speakers = [
            msg[input_message_config.role_field] for msg in conversation
        ]
        return all(
            speakers[i] != speakers[i + 1] for i in range(len(speakers) - 1)
        )

    self.dataset = self.dataset.map(
        lambda x: {output_column: _check_alternating(x[input_column])}
    )
    return self


def get_last_speaker(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    output_column: str = "__last_speaker",
):
    """
    Gets the role of the last speaker for each conversation.
    """
    if isinstance(input_message_config, dict):
        input_message_config = MessageConfig(**input_message_config)
    self.dataset = self.dataset.map(
        lambda x: {
            output_column: x[input_column][-1][input_message_config.role_field]
        }
    )
    return self


def standardize_last_turn(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    last_speaker_role: Literal["user", "assistant", "system"],
    output_column: Optional[str] = None,
):
    """
    Strip messages from the end of the conversation until the last message is from the specified speaker.
    This is often helpful for training in the case where you want the last reply to be from the assistant,
    and you want to train a model to generate that message given the conversation so far.
    """
    if isinstance(input_message_config, dict):
        input_message_config = MessageConfig(**input_message_config)

    if output_column is None:
        output_column = input_column

    def _standardize_single(conversation: list[dict]):
        expected_role = {
            "user": input_message_config.user_role,
            "assistant": input_message_config.assistant_role,
            "system": input_message_config.system_role,
        }[last_speaker_role]

        while (
            conversation[-1][input_message_config.role_field] != expected_role
        ):
            conversation = conversation[:-1]
        return conversation

    self.dataset = self.dataset.map(
        lambda x: {output_column: _standardize_single(x[input_column])}
    )
    return self


def get_shared_prefix(
    self,
    col1: str,
    col2: str,
    prefix_col: str,
    suffix1_col: Optional[str] = None,
    suffix2_col: Optional[str] = None,
    strip_whitespace: bool = True,
):
    """
    When you have "chosen" and "rejected" conversations, this method will find the longest shared prefix between them.
    Operates on strings, not lists of messages. Resulting prefix will be put in 'prefix_col'. If provided, each suffix
    will be put in the corresponding suffix columns.
    """

    def _get_prefix_single(sample: dict):
        s1, s2 = sample[col1], sample[col2]
        if strip_whitespace:
            s1, s2 = s1.strip(), s2.strip()
        prefix = os.path.commonprefix([s1, s2])
        suffix1 = s1[len(prefix) :]
        suffix2 = s2[len(prefix) :]
        result = {prefix_col: prefix}
        if suffix1_col is not None:
            result[suffix1_col] = suffix1
        if suffix2_col is not None:
            result[suffix2_col] = suffix2
        return result

    self.dataset = self.dataset.map(_get_prefix_single)
    return self


def add_initial_system_message(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    system_message: str,
    output_column: Optional[str] = None,
):
    """
    Adds (the same) initial system message to every conversation.
    """
    if isinstance(input_message_config, dict):
        input_message_config = MessageConfig(**input_message_config)
    if output_column is None:
        output_column = input_column

    new_message = [
        {
            input_message_config.role_field: input_message_config.system_role,
            input_message_config.content_field: system_message,
        }
    ]
    self.dataset.map(lambda x: {output_column: new_message + x[input_column]})
    return self


def take_initial_system_message(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    output_column: str = "__initial_system_message",
    output_type: Literal["dict", "string"] = "string",
    remove_from_input: bool = True,
):
    """
    Takes initial message and puts it in another column if it's a system message.
    If there's no initial system message, it will be [None? empty?]
    """

    def _take_system_msg_single(conversation: list[dict]):
        if conversation[0]["role"] == "system":
            return {
                output_column: conversation[0]
                if output_type == "dict"
                else conversation[0][input_message_config.content_field],
                input_column: conversation[1:]
                if remove_from_input
                else conversation,
            }
        else:
            return {output_column: None, input_column: conversation}

    self.dataset = self.dataset.map(
        lambda x: _take_system_msg_single(x[input_column])
    )
    return self


def take_last_message(
    self,
    input_column: str,
    input_message_config: Union[MessageConfig, dict],
    output_column: str,
    output_type: Literal["dict", "string"] = "string",
    remove_from_input: bool = True,
):
    """
    Take the last message from a conversation and put it in a separate column.
    It will be removed from the input if remove_from_input is True.
    """
    if output_column is None:
        output_column = input_column
    if isinstance(input_message_config, dict):
        input_message_config = MessageConfig(**input_message_config)

    def _take_last_single(conversation: list[dict]):
        return {
            output_column: conversation[-1]
            if output_type == "dict"
            else conversation[-1][input_message_config.content_field],
            input_column: conversation[:-1]
            if remove_from_input
            else conversation,
        }

    self.dataset = self.dataset.map(
        lambda x: _take_last_single(x[input_column])
    )
    return self

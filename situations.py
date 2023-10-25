




def smalltalk_at_party(name_1, name_2):
    """
    Provides a scenario in which the agent talks to an unknown person at a party
    """
    observations_agent1 = [
        f"{name_1} knows Julia from the university volleyball club",
        f"{name_1} is at a party of their friend Julia. It is a house party with about 50 guests.",
        f"{name_1} only knows Julia and two other people from their volleyball team on the party",
        f"{name_1} is in the kitchen, getting a drink and eating some snacks.",
        f"{name_1} gets into a conversation with {name_2}, while they are both mixing a drink for them selves."
    ]

    observations_agent2 = [
        f"{name_2} is visiting Julia for the weekend",
        f"{name_2} is at a party of Julia. It is a house party with about 50 guests."
        f"{name_2} only knows Julia at the party" 
        f"{name_2} is in the kitchen, getting a drink and eating some snacks.",
        f"{name_2} gets into a conversation with {name_1}, while they are both mixing a drink for them selves."
    ]
    initial_observation = f"{name_1} gets into a conversation with {name_2}"
    situation_summary = (f"{name_1} and {name_2} are at a party of a common friend. They come into conversation "
                         f"while gettings drinks in the kitchen")
    return observations_agent1, observations_agent2, initial_observation, situation_summary


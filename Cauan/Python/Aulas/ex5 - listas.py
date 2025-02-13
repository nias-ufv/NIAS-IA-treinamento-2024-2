def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    fashionable=(len(arrivals)//2)+1
    print(arrivals[fashionable:])
    if name in arrivals[fashionable:] and not name==arrivals[-1]:
        return True
    else:
        return False

guests=['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']
print(fashionably_late(guests, 'Mona'))
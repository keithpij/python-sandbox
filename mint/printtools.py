'''
Tools for printing to the console.
'''


def print_transactions(title, transactions):
    total = 0
    print('\n\n' + title + '\n')
    transactions.sort(key=lambda x: x.transaction_date)
    for transaction in transactions:
        print(pad(transaction.transaction_date, 15) + pad(transaction.description, 40) + pad(transaction.category, 20) + pad('${:9,.2f}'.format(transaction.amount),20))
        total += transaction.amount


def print_transaction_totals(title, transactions):
    total = 0
    for transaction in transactions:
        if transaction.category.lower() != 'credit card payment':
            total += transaction.amount
    print(pad(title, 20) + '\t' + '${:9,.2f}'.format(total))


def print_category_totals(categories):
    # Create a dictionary of totals for each category.
    category_totals = dict()
    for category_name in categories.keys():
        total = 0
        for transaction in categories[category_name]:
            total += transaction.amount
        category_totals[category_name] = total

    # Loop through and print the totals for each category.
    for category_name in sorted(category_totals, key=category_totals.__getitem__, reverse=True):  #sorted(d, key=d.__getitem__)
        category_total = category_totals[category_name]
        print(pad(category_name, 20) + '\t' + '${:9,.2f}'.format(category_total))


def print_category_comparison(previous_month_totals, current_month_totals):
    print(pad('Category', 30) + pad('Previous Month', 20) + pad('Current Month', 20) + pad('Difference', 20))

    for category_name in sorted(previous_month_totals, key=previous_month_totals.__getitem__, reverse=True):
        previous = previous_month_totals[category_name]
        if category_name in current_month_totals:
            current = current_month_totals[category_name]
        else:
            current = 0
        diff = previous - current
        print(pad(category_name, 30) + pad('${:9,.2f}'.format(previous), 20) + pad('${:9,.2f}'.format(current), 20) + pad('${:9,.2f}'.format(diff), 20))


def print_categories(categories):
    for category in categories:
        print_transactions(category, categories[category])


def pad(value, width):
    '''
    Turns value into a string and then pads the new string value with spaces
    to produce a string of length width.
    '''
    display = str(value)
    length = len(display)

    if length >= width:
        return display[0:width]

    delta = width - length
    for _ in range(delta):
        display = display + ' '
    return display

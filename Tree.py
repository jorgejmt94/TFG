import DB
class Node():
    key = 0

    def __init__(self, key):
        self.key = key
        #self.word = word
        self.left = None
        self.right = None


class AVLTree():
    def __init__(self, *args):
        self.node = None
        self.height = -1
        self.balance = 0

        if len(args) == 1:
            for i in args[0]:
                self.insert(i)

    def height(self):
        if self.node:
            return self.node.height
        else:
            return 0

    def is_leaf(self):
        return (self.height == 0)

    def insert(self, key):
        tree = self.node

        newnode = Node(key)

        if tree == None:
            self.node = newnode
            self.node.left = AVLTree()
            self.node.right = AVLTree()

        elif key < tree.key:
            self.node.left.insert(key)
        elif key > tree.key:
            self.node.right.insert(key)

        #else:
            #print("Word [" + str(key) + "] already in tree.")

        self.sway()

    '''
        Rebalance tree 
    '''
    def sway(self):

        # key inserted. Let's check if we're balanced
        self.update_heights(False)
        self.update_balances(False)
        while self.balance < -1 or self.balance > 1:
            if self.balance > 1:
                if self.node.left.balance < 0:
                    self.node.left.left_rotate()  # we're in case II
                    self.update_heights()
                    self.update_balances()
                self.right_rotate()
                self.update_heights()
                self.update_balances()

            if self.balance < -1:
                if self.node.right.balance > 0:
                    self.node.right.sway()  # we're in case III
                    self.update_heights()
                    self.update_balances()
                self.left_rotate()
                self.update_heights()
                self.update_balances()

    def right_rotate(self):
        # Rotate left pivoting on self
        A = self.node
        B = self.node.left.node
        T = B.right.node

        self.node = B
        B.right.node = A
        A.left.node = T

    def left_rotate(self):
        # Rotate left pivoting on self
        A = self.node
        B = self.node.right.node
        T = B.left.node

        self.node = B
        B.left.node = A
        A.right.node = T

    def update_heights(self, recurse=True):
        if not self.node == None:
            if recurse:
                if self.node.left != None:
                    self.node.left.update_heights()
                if self.node.right != None:
                    self.node.right.update_heights()
            self.height = max(self.node.left.height,
                              self.node.right.height) + 1
        else:
            self.height = -1

    def update_balances(self, recurse=True):
        if not self.node == None:
            if recurse:
                if self.node.left != None:
                    self.node.left.update_balances()
                if self.node.right != None:
                    self.node.right.update_balances()
            self.balance = self.node.left.height - self.node.right.height
        else:
            self.balance = 0

    def delete(self, key):
        if self.node != None:
            if self.node.key == key:
                if self.node.left.node == None and self.node.right.node == None:
                    self.node = None  # leaves can be killed at will
                # if only one subtree, take that
                elif self.node.left.node == None:
                    self.node = self.node.right.node
                elif self.node.right.node == None:
                    self.node = self.node.left.node

                # worst-case: both children present. Find logical successor
                else:
                    replacement = self.successor(self.node)
                    if replacement != None:  # sanity check
                        self.node.key = replacement.key
                        # replaced. Now delete the key from right child
                        self.node.right.delete(replacement.key)
                self.sway()
                return
            elif key < self.node.key:
                self.node.left.delete(key)
            elif key > self.node.key:
                self.node.right.delete(key)

            self.sway()
        else:
            return

    def predecessor(self, node):

        node = node.left.node
        if node != None:
            while node.right != None:
                if node.right.node == None:
                    return node
                else:
                    node = node.right.node
        return node

    def successor(self, node):

        node = node.right.node
        if node != None:  # just a sanity check

            while node.left != None:
                if node.left.node == None:
                    return node
                else:
                    node = node.left.node
        return node

    def check_balanced(self):
        if self == None or self.node == None:
            return True

        # We always need to make sure we are balanced
        self.update_heights()
        self.update_balances()
        return ((abs(self.balance) < 2) and self.node.left.check_balanced() and self.node.right.check_balanced())

    def insert_array(self, array):
        for word in array:
            self.insert(word)

    ''''
    El algoritmo
    '''''
    def find_words_in_text(self, text, word_mark):
        import Utils
        found_list = []
        text_words = text.split()
        return_value = index_text_word = 0
        empty_words_tree = AVLTree()
        empty_words_tree.insert_array(DB.GET_empty_words_from_DB())
        while index_text_word < len(text_words):
            end = 0
            aux_tree = self
            while end == 0:
                if aux_tree.node != None and index_text_word < len(text_words):
                    word_in_tree = aux_tree.node.key.split()
                    #print(word_in_tree) #word we are searching
                    j = 0
                    if len(word_in_tree) != 0:
                        if  Utils.stem(word_in_tree[j].lower()) == Utils.stem(text_words[index_text_word].lower()): # la hemos encontrado chic@s!
                            found_list.append(word_in_tree[j])
                            j = ok = end = 1
                            return_value += word_mark
                            while j < len(word_in_tree) and ok == 1:
                                if (index_text_word + j) < len(text_words):
                                    if False == empty_words_tree.find_word(word_in_tree[j]):
                                        #print('mirando: ---------------', text_words[index_text_word + j],word_in_tree[j], word_in_tree)
                                        if  Utils.stem(word_in_tree[j].lower()) == Utils.stem(text_words[index_text_word+j].lower()) :
                                            #print('+subPalabra encontrada!', word_in_tree[j])

                                            return_value += word_mark
                                        else:
                                            return_value -= word_mark
                                            #print('-subPalabra eliminada!', word_in_tree, text_words[index_text_word])
                                            ok = 0
                                else:
                                    ok = 0
                                j += 1

                        elif text_words[index_text_word] < word_in_tree[0]:
                            aux_tree = aux_tree.node.left
                        elif text_words[index_text_word] > word_in_tree[0]:
                            aux_tree = aux_tree.node.right
                    else:
                        end = 1
                else:
                    end = 1
            index_text_word+=1
        return return_value, found_list

    ''''
    Encontrar una palabra
    '''''
    def find_word(self, word_to_find):
        end= 0
        aux_tree = self
        while end == 0:
            if aux_tree.node != None :
                word_in_tree = aux_tree.node.key
                if  word_in_tree == word_to_find: # la hemos encontrado chic@s!
                    #print(word_to_find,'<------------')
                    return True
                elif word_to_find < word_in_tree:
                    aux_tree = aux_tree.node.left
                elif word_to_find > word_in_tree:
                    aux_tree = aux_tree.node.right
            else:
                end = 1

        return False



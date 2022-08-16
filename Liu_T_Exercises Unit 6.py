from random import choice, random
from copy import deepcopy


class PhyloNode():
    """A node on a phylogenetic tree"""

    def __init__(self, children=None, parent=None, \
                 name=None):
        """Initiate a node on a phylogenetic tree
        children -- a list of PhyloNode objects
          descending immediately from this node
        name -- a string for the name of the node
        parent -- a single PhyloNode object for the parent
        """
        self.Name = name
        self.Children = []
        if children:
            self.Children.extend(children)
        self.Parent = parent
        self.Extinct = False

    def isTip(self):
        """Return True if the node is a tip"""
        if not self.Children:  # capture None or []
            return True
        else:
            return False

    def isRoot(self):
        """Return True if the node is the root of the whole tree"""
        if not self.Parent:
            return True
        else:
            return False

    def getDescendants(self):
        """Return a list of PhyloNodes descending from the current node"""

        if self.isTip():
            print(self.Name, " is a tip ... returning []")
            return []

        descendants = self.Children or []
        for c in self.Children:
            # The set of descendants is described
            # by the descendants of all the nodes
            # immediate children.
            if not c.isTip():
                child_descendants = c.getDescendants()
                descendants.extend([c for c in child_descendants if c not in descendants])

            # Side note: this will fail on enormous trees
            # due to the recursion limit. Not normally a problem though.
        return descendants

    def getAncestors(self):
        """Return the ancestors of the given node"""
        if self.isRoot():
            return None

        ancestors = [self.Parent]
        parents_ancestors = self.Parent.getAncestors()
        if parents_ancestors:
            ancestors.extend(parents_ancestors)
        return ancestors

    def getRoot(self):
        """Return the root node"""
        curr_node = self
        while not curr_node.isRoot():
            curr_node = curr_node.Parent
        return curr_node

    def addChild(self, child):
        """Attach a child node"""
        if child not in self.Children:
            self.Children.append(child)
        child.Parent = self

    def addParent(self, parent):
        """Attach a parent node"""
        self.Parent = parent
        parent.Children.append(self)

    def getMRCA(self, other):
        """Return PhyloNode for the most recent common ancestor with other
        other -- another PhyloNode object in the same tree
        """
        most_recent_common_ancestor = None
        # Cache the ancestors of other since
        # we'll refer to it often
        other_ancestors = other.getAncestors()
        self_ancestors = self.getAncestors()
        # First check for the trivial case
        # where one node is root (but be sure they're on the same tree)
        if self.isRoot() and self in other_ancestors:
            return self

        if other.isRoot() and other in self_ancestors:
            return other

        for a in self_ancestors:
            # Note these will be in order of relatedness
            if a in other_ancestors:
                # End the loop the first time this happens
                # since we want the most
                # recent common ancestor
                most_recent_common_ancestor = a
                break

        if not most_recent_common_ancestor:
            raise ValueError("No common ancestor found for ", self.Name, " and ", other.Name, \
                             ". Are they on the same tree?")

        return most_recent_common_ancestor

    def speciate(self):
        """Add two children descending from this node"""
        if self.Children:
            raise ValueError("Internal nodes can't speciate")
        if self.Extinct:
            raise ValueError("Extinct nodes can't speciate")

        child1_name = self.Name + "A"
        child1 = PhyloNode(name=child1_name)
        self.addChild(child1)
        child2_name = self.Name + "B"
        child2 = PhyloNode(name=child2_name)
        self.addChild(child2)

    def update(self, speciation_chance=0.25,
               extinction_chance=0.25):
        """Update the node by speciation or extinction"""
        if not self.isTip():
            print("Not updating node {name} - it's not a tip".format(name=self.Name))
            return None

        if random() < extinction_chance:
            print(self.Name, " goes extinct!")
            self.Extinct = True

        if self.Extinct:
            print("Not updating node {name} - it's extinct".format(name=self.Name))
            return None
        print("Updating node:{name}".format(name=self.Name))

        if random() < speciation_chance:
            # Speciate
            print("Node {name} speciates!".format(name=self.Name))
            self.speciate()

        # Exercise 4:

    def getNodeByName(self, name):
        """
        Returns a list of nodes that have the name passed.
        If there are more than one match then the list will have a len > 1
        :param name: str name of node being searched for.
        :return: list of Phylonodes that have the specified name.
        """
        # No tree object, need to ensure same starting point for tree traversal
        curr_node = getRoot(self)

        # Check if user looking for root:
        if curr_node.Name == name:
            return curr_node

        # Get list of descendants
        descendants = curr_node.getDescendants()
        matches = []
        # Iterate through all descendants
        for x in descendants:
            # If descendants name matches add it to the list.
            if x.Name == name:
                matches.append(x)

        return matches

    # Exercise 5:
    def getSister(self):
        """
        Returns a list of PhyloNodes that are sister taxa to the curr_node
        :return: list of PhyloNodes
        """
        # If root node raise error.
        if self.isRoot():
            raise ValueError('Root node has no sister taxa.')

        # Go up to parent and get descendants
        parent_node = self.Parent
        parent_descendants = parent_node.getDescendants()
        # If descendants list is only 1 (meaning curr_node / self is only descendant)
        # move up to next ancestor
        while len(parent_descendants) == 1:
            parent_node = parent_node.Parent
            parent_descendants = parent_node.getDescendants()

        # Remove self from the list and return.
        parent_descendants.remove(self)
        return parent_descendants




if __name__ == '__main__':
    # Build our simple tree
    root = PhyloNode(name="root")
    A = PhyloNode(name="A", parent=root)
    root.addChild(A)
    B = PhyloNode(name="B", parent=root)
    root.addChild(B)
    B1 = PhyloNode(name="B1", parent=B)
    B2 = PhyloNode(name="B2", parent=B)
    B.addChild(B1)
    B.addChild(B2)

    A_B1_MRCA = B1.getMRCA(A)
    B1_B2_MRCA = B1.getMRCA(B2)

    print("The MRCA of B1 and B2 is:", B1_B2_MRCA.Name)
    print("The MRCA of A and B1 is:", A_B1_MRCA.Name)

    # Exercise 1:
    ex_1_root = PhyloNode(name='root')
    ex_1_root_l1 = PhyloNode(name='root_l1')
    ex_1_root_l2 = PhyloNode(name='root_l2')
    ex_1_root_l3 = PhyloNode(name='root_l3')
    ex_1_fish = PhyloNode(name='fish')
    ex_1_frog = PhyloNode(name='frog')
    ex_1_lizard = PhyloNode(name='lizard')
    ex_1_mouse = PhyloNode(name='mouse')
    ex_1_human = PhyloNode(name='human')

    # Build first layer of tree
    ex_1_root.addChild(ex_1_fish)
    ex_1_root.addChild(ex_1_root_l1)

    # Second layer
    ex_1_root_l1.addChild(ex_1_frog)
    ex_1_root_l1.addChild(ex_1_root_l2)

    # Third
    ex_1_root_l2.addChild(ex_1_lizard)
    ex_1_root_l2.addChild(ex_1_root_l3)

    # Last
    ex_1_root_l3.addChild(ex_1_mouse)
    ex_1_root_l3.addChild(ex_1_human)

    """
    Exercise 2:
                                    root
                                /           \
                            root_l1         fish
                           /       \
                       root_l2      frog
                    /           \
                root_l3         lizard
              /         \
            human       mouse
      
    The frog is more closely related to humans as they have a more recent
    common ancestor (root_l1) compared to humans and fish (root).
    """
    human_mrca_frog = ex_1_root_l3.getMRCA(ex_1_frog)
    print(f"Human MRCA with Frog: {human_mrca_frog.Name}")
    human_mrca_fish = ex_1_root_l3.getMRCA(ex_1_fish)
    print(f"Human MRCA with Fish: {human_mrca_fish.Name}")


    # Exercise 3:
    def getRoot(self):
        """
        Return the root node of the tree
        """
        curr_node = self

        # While isRoot returns false keep moving upward on the tree
        while not curr_node.isRoot():
            # Set current node to the parent and check if root again.
            curr_node = curr_node.Parent

            # If reach here, then node is root.
        return curr_node


    # Exercise 4:
    def getNodeByName(self, name):
        """
        Returns a list of nodes that have the name passed.
        If there are more than one match then the list will have a len > 1
        :param name: str name of node being searched for.
        :return: list of Phylonodes that have the specified name.
        """
        # No tree object, need to ensure same starting point for tree traversal
        curr_node = getRoot(self)

        # Check if user looking for root:
        if curr_node.Name == name:
            return curr_node

        # Get list of descendants
        descendants = curr_node.getDescendants()
        matches = []
        # Iterate through all descendants
        for x in descendants:
            # If descendants name matches add it to the list.
            if x.Name == name:
                matches.append(x)

        return matches

    # Test code for exercise 4
    get_node_name = ex_1_mouse.getNodeByName('frog')
    for i in get_node_name:
        print(i.Name)

    # Exercise 5:
    def getSister(self):
        """
        Returns a list of PhyloNodes that are sister taxa to the curr_node
        :return: list of PhyloNodes
        """
        # If root node raise error.
        if self.isRoot():
            raise ValueError('Root node has no sister taxa.')

        # Go up to parent and get descendants
        parent_node = self.Parent
        parent_descendants = parent_node.getDescendants()
        # If descendants list is only 1 (meaning curr_node / self is only descendant)
        # move up to next ancestor
        while len(parent_descendants) == 1:
            parent_node = parent_node.Parent
            parent_descendants = parent_node.getDescendants()

        # Remove self from the list and return.
        parent_descendants.remove(self)
        return parent_descendants

    # Test code for exercise 5
    ex_5_sister_test = ex_1_mouse.getSister()
    for x in ex_5_sister_test:
        print(x.Name)
# %%

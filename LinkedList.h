
class Node
{
    public:
        int data;
        Node *next;

        Node(){}
        Node(int d): data(d),next(nullptr) {}
        Node(int d,Node*next): data(d),next(next) {}
};



class LinkedList
{
    private:
        Node *HEAD;
        Node *TAIL;
    
    public:
        LinkedList();
       

        void add(Node *pred,int data);
        void addToHead(int data);
        void addToTail(int data);
        void Traverse();
        bool remove(int &data);
        bool removeFromHead(int &data);
        bool removeFromTail(int &data);
        bool isEmpty();

};

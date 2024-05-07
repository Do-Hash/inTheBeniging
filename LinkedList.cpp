#include "LinkedList.h"
#include<iostream>

LinkedList::LinkedList() : HEAD(nullptr),TAIL(nullptr) {}

bool LinkedList::isEmpty() {
    return HEAD == nullptr;
}

void LinkedList::add(Node *pred,int data)
{
    Node *newNode = new Node; // Initialize newNode pointer
    newNode->data = data;
    newNode->next = pred->next;
    pred->next = newNode;

}

void LinkedList::addToHead(int data)
{
    Node *newNode = new Node; // Initialize newNode pointer
    if(!isEmpty())
    {
        
        newNode->data= data;
        newNode->next= HEAD;
        HEAD = newNode;
    }
    else
    {
        newNode->data = data;
        newNode->next = nullptr;
        HEAD = newNode;
        TAIL=HEAD;
    }

}

void LinkedList::addToTail(int data)
{
     Node *newNode = new Node; // Initialize newNode pointer
    if(!isEmpty())
    {
        
        newNode->data= data;
        newNode->next= nullptr;
        TAIL->next = newNode;
        TAIL = newNode;
    }
    else
    {
        newNode->data = data;
        newNode->next = nullptr;
        TAIL = newNode;
        HEAD=TAIL;
    }

}

bool LinkedList::removeFromHead(int &data) {
    if (!isEmpty()) {
        Node *nodeToDelete = HEAD;
        data = HEAD->data;
        HEAD = HEAD->next;
        delete nodeToDelete;
        if (isEmpty()) {
            TAIL = nullptr;
        }
        return true;
    } else {
        return false;
    }
}

bool LinkedList::removeFromTail(int &data)
{
    if (isEmpty())
    {
        return false;
    }

    if (HEAD == TAIL)
    { // Only one node in the list
        data = HEAD->data;
        delete HEAD;
        HEAD = TAIL = nullptr;
        return true;
    }

    Node *p = HEAD;
    Node *t = nullptr;
    while (p != TAIL)
    {
        t = p;
        p = p->next;
    }

    data = TAIL->data;
    delete TAIL;
    TAIL = t;
    TAIL->next = nullptr;
    return true;
}

void LinkedList::Traverse()
{
    Node *p = HEAD;
    if (!isEmpty())
    {
        std::cout << "The contents of the list are :  ";
        while (p != NULL)
        {
            std::cout << p->data ;
            std::cout<<"  ";
            p = p->next;
        }
    }
}

bool LinkedList::remove(int &data) {
    if (isEmpty()) {
        return false;
    }
    
    if (HEAD->data == data) {
        return removeFromHead(data);
    }
    
    Node *p = HEAD;
    Node *temp = nullptr;
    while (p->next != nullptr && p->data != data) {
        temp = p;
        p = p->next;
    }
    
    if (p->data == data) {
        if (p == TAIL) {
            return removeFromTail(data);
        } else {
            Node *nodeToDelete = p;
            temp->next = p->next;
            delete nodeToDelete;
            return true;
        }
    }
    
    return false;
}
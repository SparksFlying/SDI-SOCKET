#include"entity.hpp"
#include<array>
using Entity::record;
using std::array;

int main(){
    vector<record<2>> data(1);
    data[0].id = 0;
    data[0].key = array<uint32_t, 2>{1, 2};
    data[0].value  = nullptr;
    Entity::DO<2> DataOwner(data);
    DataOwner.buildIndex();
}
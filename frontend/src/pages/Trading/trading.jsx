import React, { useState, useEffect } from 'react'   
import Navbar from '../Navbar/navbar';
import Form from './Form/form';
import Performance from './Performance/performance';
import Newchart from './ShowChart/newchart';
import News from '../News/news';
import Transactions from '../News/transactions';
import Models from './Models/models';
import Results from './Models/results';

const trading = () => {
  const [data, setData] = useState([{}]);
  const [formdata, setFormdata] = useState([{}]);
  const [lastTransaction, setLastTransaction] = useState([{}]);

  useEffect(() => {
    fetch(process.env.BACKEND_URL + "/data").then(response => 
      response.json().then(data_obj => {
        setData(data_obj);
        setLastTransaction(data_obj);
      })
    );
  }, [])

  const retrieveData = (formData) => {
    setFormdata(formData);
  };

  const [childData, setChildData] = useState(0);
  const [modelData, setModelData] = useState(0);

  useEffect(() => {
    setModelData(0);
  }, [formdata])

  return (
    <div className="min-h-screen overflow-auto">
      <Navbar/>
      <div className="w-auto mx-10 my-5">
        <div className="flex flex-row">
          <Form onSubmit={retrieveData}/>
          <Newchart data={formdata} model={modelData}/>
          <Models data={formdata} passModelData={setModelData} passChildData={setChildData} setLastTransaction={setLastTransaction}/>
          <Results data={formdata.ticker} />
        </div>
        <div className="flex flex-row">
          <Performance/>
          <Transactions data={childData} transaction={lastTransaction}/>
          <News/>
        </div>
      </div>
    </div>
  )
}

export default trading
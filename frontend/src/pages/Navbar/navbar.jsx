import React, { useState } from 'react';
import "./style.css";
import { useNavigate } from 'react-router-dom';

const navbar = () => {
  const [page, setPage] = useState(0)
  const navigate = useNavigate();

  return (
    <div className="bg-white p-5 inline-block min-w-[122%] max-w-[125%] font-body">
         <div className="not-italic font-bold text-2xl leading-8 flex items-center text-gray-700 float-left ml-12">
            FAANG Watcher
        </div>
        
        <div className="float-left ml-12 leading-6 flex items-center">
            <div className="not-italic font-bold text-lg leading-8 flex items-center text-indigo-700 ml-48 no-underline" onClick={() => {
              navigate('/')
            }}>
              Algo Trading
            </div>
            <div className="border rotate-0 h-5 mx-5 my-0 border-solid border-[#A9ACBB]">
                
            </div>
            <div className="not-italic font-medium text-lg leading-8 flex items-center text-gray-500" onClick={() => {
              navigate('/about')
            }}>
              About               
            </div>
        </div>
    </div>
  )
}

export default navbar
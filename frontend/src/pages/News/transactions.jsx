import React, { useState, useEffect } from 'react';

const Transactions = (props) => {
    const [data, setData] = useState(props.data);
    const [transaction, setTransaction] = useState(props.transaction);

    useEffect(() => {
        setData(props.data); 
    }, [props.data]);

    useEffect(() => {
        setTransaction(props.transaction); 
    }, [props.transaction]);

    return (
        <div className='min-w-[25%] max-w-[25%] mt-6 ml-6'>
            <div className="bg-white rounded-lg p-5">
                <div className="flex items-center">
                    <img src="src/assets/transactions.png" alt='Transactions' className='w-16' />
                    <div className="ml-auto font-dm-sans font-bold text-2xl leading-[23px] flex items-center text-gray-700">
                        {data}
                    </div>
                </div>
                <div className="mt-6 font-dm-sans font-semibold text-[19.3548px] leading-[25px] flex items-center text-gray-500">
                    Total models generated
                </div>
            </div>

            <div className="mt-6 bg-white rounded-lg p-5">
                <div className="font-dm-sans font-semibold text-lg leading-5 flex items-center text-gray-500 mb-5">
                    Previous Model
                </div>  
                <div className="flex flex-row">
                    <div className="flex-grow min-w-0"> {/* Adjusted for spacing */}
                        <div className="font-dm-sans font-medium text-xl leading-6 flex items-center text-blue-700">
                            {transaction.ticker}
                        </div>
                        <div className="font-dm-sans font-normal text-[18px] leading-5 flex items-center text-gray-400 mt-3">
                            {transaction.time}
                        </div>
                    </div>

                    <div className="ml-4 flex-shrink-0" style={{ minWidth: '100px' }}> {/* Ensure container can grow */}
                        <div className="flex flex-row justify-end">
                            <div className="font-dm-sans font-medium text-2xl leading-6 items-center text-gray-700">
                                ${transaction.price}
                            </div>
                            <div className={`text-xl leading-6 items-center ${transaction.change > 0 ? 'text-[#63C89B]' : 'text-[#EB5757]'} ml-2`}>
                                {transaction.change}%
                            </div>
                        </div>
                        <div className="font-dm-sans font-normal text-[18px] leading-5 flex items-center text-gray-700 mt-3">
                            {transaction.modal}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Transactions;

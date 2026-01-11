import {getRequestConfig} from 'next-intl/server';
 
export default getRequestConfig(async () => {
  // Provide a static locale for now
  const locale = 'cs';
 
  return {
    locale,
    messages: (await import(`../locale/${locale}.json`)).default,
    timeZone: 'Europe/Prague'
  };
});

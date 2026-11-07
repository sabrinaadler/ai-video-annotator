const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const base=process.env.FIELDNOTE_TEST_URL||'http://127.0.0.1:8766';
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const page=await browser.newPage({viewport:{width:1440,height:1080}});page.setDefaultTimeout(15000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(base);await page.waitForFunction(()=>document.querySelector('#save').textContent==='Session ready');
await page.waitForFunction(()=>document.querySelector('video').readyState>=1);console.log('metadata loaded');
assert(await page.evaluate(()=>document.querySelector('video').videoWidth)>0);
await page.locator('#play').click();await page.waitForFunction(()=>document.querySelector('video').currentTime>.25);await page.locator('#play').click();
for(const [start,end] of [[48.9666667,50.0666667],[88,88.6],[142.1,142.6666667],[206.4666667,206.8333333],[518.1666667,520.3]]){
await page.locator('#start').fill(String(start));await page.locator('#end').fill(String(end));await page.locator('#add').click();await page.waitForFunction(()=>document.querySelector('#save').textContent==='Saved to disk');
}
console.log('examples marked');assert.equal(await page.locator('#examples li').count(),5);
await page.locator('#search').click();await page.waitForFunction(()=>document.querySelectorAll('.candidate').length===20);
assert((await page.locator('.candidate strong').first().innerText()).startsWith('03:23.000'));
await page.waitForFunction(()=>document.querySelector('video').currentTime>200);
await page.locator('#start').fill('203.1');await page.locator('#end').fill('204.9');
await page.locator('#note').fill('=test, "quoted"\nsecond line');await page.locator('#accept').click();
await page.waitForFunction(()=>document.querySelector('#summary').textContent.startsWith('1 accepted'));
await page.locator('#later').click();await page.waitForFunction(()=>document.querySelectorAll('.candidate.later').length===1);
await page.locator('#reject').click();await page.waitForFunction(()=>document.querySelectorAll('.candidate.rejected').length===1);
await page.locator('#undo').click();await page.waitForFunction(()=>document.querySelectorAll('.candidate.rejected').length===0);
let session=await (await page.request.get(base+'/api/export.json')).json();
assert.equal(session.results[0].start,203.1);assert.equal(session.results[0].status,'accepted');assert.equal(session.results[0].original_start,203);assert.equal(session.results.filter(r=>r.status==='later').length,1);
const csv=await (await page.request.get(base+'/api/export.csv')).text();assert(csv.includes("'=test"));assert(csv.includes('203.1,204.9'));assert(csv.includes('""quoted""'));
await page.reload();await page.waitForFunction(()=>document.querySelector('#summary').textContent.startsWith('1 accepted'));console.log('examples marked');assert.equal(await page.locator('#examples li').count(),5);
await page.screenshot({path:'/tmp/fieldnote-working.png',fullPage:true});
for(const width of [1024,736,360]){await page.setViewportSize({width,height:1000});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`overflow at ${width}`);}
const downloadPromise=page.waitForEvent('download');await page.locator('a[download]').first().click();const download=await downloadPromise;assert.equal(download.suggestedFilename(),'annotations.csv');assert.equal(await download.failure(),null);
assert.deepEqual(errors,[]);console.log('Real MP4 playback, actual retrieval, decisions, undo, persistence, exports and responsive layout passed.');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
